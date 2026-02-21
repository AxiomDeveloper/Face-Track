import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';

const { FaceDetector, FilesetResolver } = vision;

const STORAGE_KEY = 'facesnap-captures-v1';

const dom = {
  video: document.getElementById('camera'),
  overlay: document.getElementById('overlay'),
  wrap: document.getElementById('videoWrap'),
  startBtn: document.getElementById('startBtn'),
  flipBtn: document.getElementById('flipBtn'),
  clearBtn: document.getElementById('clearBtn'),
  gallery: document.getElementById('gallery'),
  template: document.getElementById('captureTemplate')
};

const ctx = dom.overlay.getContext('2d');

let faceDetector;
let stream;
let running = false;
let facingMode = 'user';
let currentDetections = [];
let rafId;

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

    node
      .querySelector('[data-action="save"]')
      .addEventListener('click', () => saveOrShare(item, index));

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
  } catch {
    // Ignore and fallback to download.
  }

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

  for (let i = 0; i < bytes.length; i += 1) {
    buffer[i] = bytes.charCodeAt(i);
  }

  return new File([buffer], fileName, { type: mime });
}

function waitForVideoReady(videoEl) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Video metadata timeout')), 8000);

    const done = () => {
      clearTimeout(timeout);
      videoEl.removeEventListener('loadedmetadata', done);
      resolve();
    };

    if (videoEl.readyState >= 1 && videoEl.videoWidth > 0) {
      clearTimeout(timeout);
      resolve();
      return;
    }

    videoEl.addEventListener('loadedmetadata', done, { once: true });
  });
}

async function safeGetUserMedia(preferredFacingMode) {
  // iOS Safari can throw OverconstrainedError on "ideal width/height" + facingMode combos.
  // Keep constraints simple and progressively fallback.
  const attempts = [
    // Try facingMode as string (often best on iOS)
    { video: { facingMode: preferredFacingMode }, audio: false },

    // Try ideal facingMode object
    { video: { facingMode: { ideal: preferredFacingMode } }, audio: false },

    // Final fallback: just request any camera
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
  if (faceDetector) return;

  const resolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  faceDetector = await FaceDetector.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
    },
    runningMode: 'VIDEO',
    minDetectionConfidence: 0.55
  });
}

async function startCamera() {
  if (!window.isSecureContext) {
    throw new Error('Camera requires HTTPS (secure context).');
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia not available. Use Safari (not in-app browser) and update iOS if needed.');
  }

  await initDetector();
  stopCamera();

  // iOS: set these in JS as well
  dom.video.setAttribute('playsinline', '');
  dom.video.playsInline = true;
  dom.video.muted = true;

  stream = await safeGetUserMedia(facingMode);

  dom.video.srcObject = stream;

  // iOS: wait for metadata before play()
  await waitForVideoReady(dom.video);

  try {
    await dom.video.play();
  } catch (e) {
    // Stream can still be active even if play() complains.
    console.warn('video.play() failed:', e);
  }

  resizeOverlay();
  running = true;
  dom.flipBtn.disabled = false;
  detectLoop();
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  // Helps iOS release camera cleanly
  try {
    dom.video.pause?.();
  } catch {}
  dom.video.srcObject = null;
}

function resizeOverlay() {
  const rect = dom.wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  dom.overlay.width = Math.floor(rect.width * dpr);
  dom.overlay.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawFaces() {
  const width = dom.wrap.clientWidth;
  const height = dom.wrap.clientHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#4cc2ff';
  ctx.fillStyle = 'rgba(76,194,255,.12)';

  for (const face of currentDetections) {
    const { originX, originY, width: w, height: h } = face.boundingBox;
    ctx.fillRect(originX, originY, w, h);
    ctx.strokeRect(originX, originY, w, h);
  }
}

async function detectLoop() {
  if (!running || !faceDetector || dom.video.readyState < 2) {
    rafId = requestAnimationFrame(detectLoop);
    return;
  }

  try {
    const result = faceDetector.detectForVideo(dom.video, performance.now());
    currentDetections = result.detections || [];
    drawFaces();
  } catch (e) {
    // If something fails mid-stream, don't brick the loop.
    console.warn('detectForVideo failed:', e);
  }

  rafId = requestAnimationFrame(detectLoop);
}

function captureFace(face) {
  const source = document.createElement('canvas');
  source.width = dom.video.videoWidth;
  source.height = dom.video.videoHeight;
  const sctx = source.getContext('2d');
  sctx.drawImage(dom.video, 0, 0, source.width, source.height);

  const scaleX = source.width / dom.wrap.clientWidth;
  const scaleY = source.height / dom.wrap.clientHeight;
  const pad = 22;

  const x = Math.max(0, (face.boundingBox.originX - pad) * scaleX);
  const y = Math.max(0, (face.boundingBox.originY - pad) * scaleY);
  const w = Math.min(source.width - x, (face.boundingBox.width + pad * 2) * scaleX);
  const h = Math.min(source.height - y, (face.boundingBox.height + pad * 2) * scaleY);

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

  return currentDetections.find((face) => {
    const b = face.boundingBox;
    return x >= b.originX && x <= b.originX + b.width && y >= b.originY && y <= b.originY + b.height;
  });
}

dom.startBtn.addEventListener('click', async () => {
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    alert(
      `Unable to start camera.\n\n` +
        `Name: ${error?.name || 'Unknown'}\n` +
        `Message: ${error?.message || error}\n\n` +
        `Fix tips:\n` +
        `• iOS Settings > Safari > Camera = Allow\n` +
        `• Open in Safari (not inside X/IG/TikTok in-app browser)\n` +
        `• Refresh twice after deploying (service worker cache)\n`
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