import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';

const STORAGE_KEY = 'facesnap-captures-v1';

const dom = {
  video: document.getElementById('camera'),
  overlay: document.getElementById('overlay'),
  wrap: document.getElementById('videoWrap'),
  startBtn: document.getElementById('startBtn'),
  flipBtn: document.getElementById('flipBtn'),
  clearBtn: document.getElementById('clearBtn'),
  gallery: document.getElementById('gallery'),
  template: document.getElementById('captureTemplate'),
  statusLine: document.getElementById('statusLine'),
  iosBtn: document.getElementById('iosBtn'),
  iosCapture: document.getElementById('iosCapture')
};

const ctx = dom.overlay.getContext('2d');

let faceLandmarker;
let stream;
let running = false;
let facingMode = 'user';
let currentDetections = [];
let rafId;

function setStatus(text) {
  dom.statusLine.textContent = `Status: ${text}`;
}

function isInAppBrowser() {
  const ua = navigator.userAgent || '';
  return /Instagram|FBAN|FBAV|FB_IAB|Line|TikTok|Twitter|X\/|Snapchat/.test(ua);
}

function loadCaptures() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveCaptures(captures) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(captures));
}

function renderGallery() {
  const captures = loadCaptures();
  dom.gallery.innerHTML = '';
  captures.forEach((item, index) => {
    const node = dom.template.content.firstElementChild.cloneNode(true);
    const img = node.querySelector('img');
    img.src = item;

    node.querySelector('[data-action="save"]').addEventListener('click', () => saveOrShare(item, index));
    node.querySelector('[data-action="delete"]').addEventListener('click', () => {
      const next = loadCaptures();
      next.splice(index, 1);
      saveCaptures(next);
      renderGallery();
    });

    dom.gallery.appendChild(node);
  });
}

async function saveOrShare(dataUrl, index) {
  const file = dataUrlToFile(dataUrl, `facesnap-${Date.now()}-${index}.jpg`);
  const shareData = { files: [file], title: 'FaceSnap Capture' };

  try {
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share(shareData);
      return;
    }
  } catch {}

  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = file.name;
  a.click();
}

function dataUrlToFile(dataUrl, fileName) {
  const [meta, b64] = dataUrl.split(',');
  const mime = meta.match(/:(.*?);/)[1];
  const bytes = atob(b64);
  const buffer = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 1) buffer[i] = bytes.charCodeAt(i);
  return new File([buffer], fileName, { type: mime });
}

function waitForVideoReady(videoEl) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('Video metadata timeout')), 8000);

    const done = () => {
      clearTimeout(t);
      videoEl.removeEventListener('loadedmetadata', done);
      resolve();
    };

    if (videoEl.readyState >= 1 && videoEl.videoWidth > 0) {
      clearTimeout(t);
      resolve();
      return;
    }
    videoEl.addEventListener('loadedmetadata', done, { once: true });
  });
}

async function safeGetUserMedia(preferredFacingMode) {
  const attempts = [
    { video: { facingMode: preferredFacingMode }, audio: false },
    { video: { facingMode: { ideal: preferredFacingMode } }, audio: false },
    { video: true, audio: false }
  ];

  let lastErr;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('Unable to access camera');
}

async function initDetector() {
  if (faceLandmarker) return;

  setStatus('loading AI landmarker…');

  const resolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU'
    },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1
  });

  setStatus('AI ready');
}

async function startCamera() {
  if (!window.isSecureContext) throw new Error('Camera requires HTTPS (secure context).');
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia not available.');
  if (isInAppBrowser()) throw new Error('In-app browser detected. Open in Safari for live camera.');

  await initDetector();
  stopCamera();

  dom.video.setAttribute('playsinline', '');
  dom.video.playsInline = true;
  dom.video.muted = true;
  dom.video.autoplay = true;

  setStatus('requesting camera permission…');

  stream = await safeGetUserMedia(facingMode);
  dom.video.srcObject = stream;

  try {
    await dom.video.play();
  } catch (e) {
    console.warn('Initial play failed, waiting for metadata:', e);
  }

  await waitForVideoReady(dom.video);

  resizeOverlay();
  running = true;
  dom.flipBtn.disabled = false;
  setStatus('live AI tracking running');
  detectLoop();
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  try { dom.video.pause?.(); } catch {}
  dom.video.srcObject = null;
}

function resizeOverlay() {
  const rect = dom.wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  dom.overlay.width = Math.floor(rect.width * dpr);
  dom.overlay.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Helper to convert the 478 mesh points into a clean bounding box
function getBoundingBox(landmarks) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const pt of landmarks) {
    if (pt.x < minX) minX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y > maxY) maxY = pt.y;
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function drawFaces() {
  const width = dom.wrap.clientWidth;
  const height = dom.wrap.clientHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#4cc2ff';

  for (const faceLandmarks of currentDetections) {
    for (const point of faceLandmarks) {
      const x = point.x * width;
      const y = point.y * height;
      
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function detectLoop() {
  if (!running || !faceLandmarker || dom.video.readyState < 2 || dom.video.videoWidth === 0) {
    rafId = requestAnimationFrame(detectLoop);
    return;
  }

  try {
    const result = faceLandmarker.detectForVideo(dom.video, performance.now());
    currentDetections = result.faceLandmarks || [];
    drawFaces();
  } catch (e) {
    console.warn('AI detection failed:', e);
  }

  rafId = requestAnimationFrame(detectLoop);
}

function captureFace(landmarks) {
  const vw = dom.video.videoWidth;
  const vh = dom.video.videoHeight;
  const box = getBoundingBox(landmarks);

  const source = document.createElement('canvas');
  source.width = vw;
  source.height = vh;
  source.getContext('2d').drawImage(dom.video, 0, 0, vw, vh);

  // Translate normalized 0..1 coordinates to actual video pixels
  const originX = box.minX * vw;
  const originY = box.minY * vh;
  const width = box.width * vw;
  const height = box.height * vh;
  const pad = Math.max(vw, vh) * 0.08; // 8% padding around the face

  const x = Math.max(0, originX - pad);
  const y = Math.max(0, originY - pad);
  const w = Math.min(vw - x, width + pad * 2);
  const h = Math.min(vh - y, height + pad * 2);

  const crop = document.createElement('canvas');
  crop.width = Math.max(1, Math.round(w));
  crop.height = Math.max(1, Math.round(h));
  crop.getContext('2d').drawImage(source, x, y, w, h, 0, 0, crop.width, crop.height);

  const dataUrl = crop.toDataURL('image/jpeg', 0.92);
  const captures = loadCaptures();
  captures.unshift(dataUrl);
  saveCaptures(captures.slice(0, 40));
  renderGallery();
}

function pickFaceAtPoint(clientX, clientY) {
  const rect = dom.wrap.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;

  return currentDetections.find((landmarks) => {
    const box = getBoundingBox(landmarks);
    const pX = box.minX * rect.width;
    const pY = box.minY * rect.height;
    const pW = box.width * rect.width;
    const pH = box.height * rect.height;
    
    return x >= pX && x <= pX + pW && y >= pY && y <= pY + pH;
  });
}

function handleIOSPhoto(file) {
  if (!file) return;

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    const captures = loadCaptures();
    captures.unshift(dataUrl);
    saveCaptures(captures.slice(0, 40));
    renderGallery();

    URL.revokeObjectURL(url);
    setStatus('photo captured (not live)');
  };
  img.src = url;
}

dom.startBtn.addEventListener('click', async () => {
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    setStatus(`live failed: ${error?.name || 'Error'} / ${error?.message || error}`);
    alert(
      `Live Camera failed.\n\nName: ${error?.name || 'Unknown'}\nMessage: ${error?.message || error}\n\nTry:\n• Open in Safari (not IG/X/TikTok browser)\n• iOS Settings > Safari > Camera = Allow`
    );
  }
});

dom.flipBtn.addEventListener('click', async () => {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    alert(`Unable to flip camera:\n${error?.name || ''}\n${error?.message || error}`);
  }
});

dom.iosBtn.addEventListener('click', () => {
  dom.iosCapture.click();
});

dom.iosCapture.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  handleIOSPhoto(file);
  e.target.value = '';
});

dom.clearBtn.addEventListener('click', () => {
  saveCaptures([]);
  renderGallery();
});

dom.overlay.addEventListener('click', (event) => {
  const hit = pickFaceAtPoint(event.clientX, event.clientY);
  if (hit) captureFace(hit);
});

window.addEventListener('resize', resizeOverlay);
window.addEventListener('beforeunload', stopCamera);

renderGallery();
setStatus('ready');
