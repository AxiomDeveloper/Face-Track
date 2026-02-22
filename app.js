import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm';

const STORAGE_KEY = 'facesnap-tactical-v6';

const dom = {
  video: document.getElementById('camera'),
  overlay: document.getElementById('overlay'),
  wrap: document.getElementById('videoWrap'),
  startBtn: document.getElementById('startBtn'),
  flipBtn: document.getElementById('flipBtn'),
  clearBtn: document.getElementById('clearBtn'),
  saveAllBtn: document.getElementById('saveAllBtn'),
  zoomSlider: document.getElementById('zoomSlider'),
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
let facingMode = 'environment';
let currentDetections = [];
let currentBlendshapes = [];
let rafId;

function setStatus(text) {
  dom.statusLine.textContent = `Status: ${text}`;
}

function loadCaptures() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } 
  catch { return []; }
}

function saveCaptures(captures) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(captures));
}

function renderGallery() {
  const captures = loadCaptures();
  dom.gallery.innerHTML = '';
  
  dom.saveAllBtn.style.display = captures.length > 0 ? 'flex' : 'none';

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

function dataUrlToFile(dataUrl, fileName) {
  const [meta, b64] = dataUrl.split(',');
  const mime = meta.match(/:(.*?);/)[1];
  const bytes = atob(b64);
  const buffer = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 1) buffer[i] = bytes.charCodeAt(i);
  return new File([buffer], fileName, { type: mime });
}

async function saveOrShare(dataUrl, index) {
  const file = dataUrlToFile(dataUrl, `target-${Date.now()}-${index}.jpg`);
  try {
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: 'Target Extracted' });
      return;
    }
  } catch {}
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = file.name;
  a.click();
}

async function saveAllToDevice() {
  const captures = loadCaptures();
  if (captures.length === 0) return;
  
  const files = captures.map((dataUrl, i) => dataUrlToFile(dataUrl, `target-${Date.now()}-${i}.jpg`));
  
  try {
    if (navigator.canShare && navigator.canShare({ files })) {
      await navigator.share({ files, title: 'Tactical Extraction' });
      return;
    }
  } catch (e) { console.log('Share API failed, falling back to individual downloads'); }

  files.forEach(file => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(file);
    a.download = file.name;
    a.click();
  });
}

function waitForVideoReady(videoEl) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('Video timeout')), 8000);
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
  // Demand Maximum Resolution for long-distance targeting
  const attempts = [
    { 
      video: { 
        facingMode: { ideal: preferredFacingMode },
        width: { ideal: 3840 }, 
        height: { ideal: 2160 }
      }, 
      audio: false 
    },
    { 
      video: { 
        facingMode: { ideal: preferredFacingMode },
        width: { ideal: 1920 }, 
        height: { ideal: 1080 }
      }, 
      audio: false 
    },
    { video: { facingMode: preferredFacingMode }, audio: false }
  ];
  let lastErr;
  for (const constraints of attempts) {
    try { return await navigator.mediaDevices.getUserMedia(constraints); } 
    catch (e) { lastErr = e; }
  }
  throw lastErr || new Error('Unable to access high-res camera');
}

async function initDetector() {
  if (faceLandmarker) return;
  setStatus('Loading AI core...');
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
    numFaces: 20, 
    // Aggressive Confidence Drops for distant faces
    minFaceDetectionConfidence: 0.25, 
    minFacePresenceConfidence: 0.25,
    minTrackingConfidence: 0.25
  });
  setStatus('AI Ready');
}

async function startCamera() {
  if (!window.isSecureContext) throw new Error('Requires HTTPS.');
  await initDetector();
  stopCamera();

  dom.video.setAttribute('playsinline', '');
  dom.video.playsInline = true;
  dom.video.muted = true;
  dom.video.autoplay = true;

  setStatus('Requesting high-res optics...');
  stream = await safeGetUserMedia(facingMode);
  dom.video.srcObject = stream;

  try { await dom.video.play(); } catch (e) {}
  await waitForVideoReady(dom.video);

  resizeOverlay();
  running = true;
  dom.flipBtn.disabled = false;

  // Zoom Logic Integration
  const track = stream.getVideoTracks()[0];
  const capabilities = track.getCapabilities ? track.getCapabilities() : {};
  
  dom.zoomSlider.oninput = (e) => {
    const val = parseFloat(e.target.value);
    if (capabilities.zoom) {
      track.applyConstraints({ advanced: [{ zoom: val }] }).catch(() => {
        dom.wrap.style.transform = `scale(${val})`;
      });
    } else {
      dom.wrap.style.transform = `scale(${val})`;
    }
  };
  
  dom.zoomSlider.value = 1;
  dom.wrap.style.transform = 'scale(1)';

  const settings = track.getSettings();
  setStatus(`Scanner Active (${settings.width}x${settings.height})`);
  
  detectLoop();
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  try { dom.video.pause?.(); } catch {}
  dom.video.srcObject = null;
  ctx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
}

function resizeOverlay() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const dpr = window.devicePixelRatio || 1;
  
  dom.wrap.style.width = `${width}px`;
  dom.wrap.style.height = `${height}px`;

  dom.overlay.width = Math.floor(width * dpr);
  dom.overlay.height = Math.floor(height * dpr);
  
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

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
  const width = window.innerWidth;
  const height = window.innerHeight;
  ctx.clearRect(0, 0, width, height);
  
  if (currentDetections.length === 0) return;

  currentDetections.forEach((landmarks, index) => {
    const box = getBoundingBox(landmarks);
    const x = box.minX * width;
    const y = box.minY * height;
    const w = box.width * width;
    const h = box.height * height;
    const pad = 12;

    ctx.strokeStyle = '#ffee00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x - pad, y - pad, w + pad * 2, h + pad * 2);

    let attnStatus = "LOCKED";
    let vocalStatus = "SILENT";
    let exprStatus = "NEUTRAL";

    if (currentBlendshapes[index]) {
      const cats = currentBlendshapes[index].categories;
      const getScore = (name) => cats.find(c => c.categoryName === name)?.score || 0;

      const jawOpen = getScore('jawOpen');
      if (jawOpen > 0.15) vocalStatus = "ACTIVE (TALKING)";

      const smile = (getScore('mouthSmileLeft') + getScore('mouthSmileRight')) / 2;
      const squint = (getScore('eyeSquintLeft') + getScore('eyeSquintRight')) / 2;

      if (smile > 0.4) exprStatus = "SMILING";
      else if (squint > 0.4) exprStatus = "SQUINTING";

      const blinkL = getScore('eyeBlinkLeft');
      const blinkR = getScore('eyeBlinkRight');
      const lookingLeft = getScore('eyeLookOutLeft') + getScore('eyeLookInRight'); 
      const lookingRight = getScore('eyeLookInLeft') + getScore('eyeLookOutRight');

      if (blinkL > 0.5 && blinkR > 0.5) {
        attnStatus = "EYES CLOSED";
      } else if (lookingLeft > 0.6 || lookingRight > 0.6) {
        attnStatus = "DISTRACTED";
      }
    }

    ctx.fillStyle = 'rgba(255, 238, 0, 0.9)';
    const labelY = (y - pad) + h + pad * 2;
    ctx.fillRect(x - pad, labelY, w + pad * 2, 64);

    ctx.fillStyle = '#000000';
    ctx.font = 'bold 11px monospace';
    ctx.fillText(`TRGT-${index + 1} // BIO-METRICS`, x - pad + 4, labelY + 14);
    ctx.fillText(`ATTN:  ${attnStatus}`, x - pad + 4, labelY + 28);
    ctx.fillText(`VOCAL: ${vocalStatus}`, x - pad + 4, labelY + 42);
    ctx.fillText(`EXPR:  ${exprStatus}`, x - pad + 4, labelY + 56);
  });
}

function detectLoop() {
  if (!running || !faceLandmarker || dom.video.readyState < 2 || dom.video.videoWidth === 0) {
    rafId = requestAnimationFrame(detectLoop);
    return;
  }
  try {
    const result = faceLandmarker.detectForVideo(dom.video, performance.now());
    currentDetections = result.faceLandmarks || [];
    currentBlendshapes = result.faceBlendshapes || [];
    drawFaces();
  } catch (e) { console.warn('AI failed:', e); }
  rafId = requestAnimationFrame(detectLoop);
}

function pickFaceAtPoint(clientX, clientY) {
  const rect = dom.wrap.getBoundingClientRect();
  const tapX = clientX - rect.left;
  const tapY = clientY - rect.top;

  return currentDetections.find((landmarks) => {
    const box = getBoundingBox(landmarks);
    const bX = box.minX * rect.width;
    const bY = box.minY * rect.height;
    const bW = box.width * rect.width;
    const bH = box.height * rect.height;
    const pad = 30; 
    return tapX >= (bX - pad) && tapX <= (bX + bW + pad) && 
           tapY >= (bY - pad) && tapY <= (bY + bH + pad);
  });
}

function captureFace(landmarks) {
  const vw = dom.video.videoWidth;
  const vh = dom.video.videoHeight;
  const box = getBoundingBox(landmarks);

  const source = document.createElement('canvas');
  source.width = vw;
  source.height = vh;
  source.getContext('2d').drawImage(dom.video, 0, 0, vw, vh);

  const originX = box.minX * vw;
  const originY = box.minY * vh;
  const width = box.width * vw;
  const height = box.height * vh;
  const pad = Math.max(vw, vh) * 0.12;

  const x = Math.max(0, originX - pad);
  const y = Math.max(0, originY - pad);
  const w = Math.min(vw - x, width + pad * 2);
  const h = Math.min(vh - y, height + pad * 2);

  const crop = document.createElement('canvas');
  crop.width = Math.max(1, Math.round(w));
  crop.height = Math.max(1, Math.round(h));
  crop.getContext('2d').drawImage(source, x, y, w, h, 0, 0, crop.width, crop.height);

  const captures = loadCaptures();
  captures.unshift(crop.toDataURL('image/jpeg', 0.95));
  saveCaptures(captures.slice(0, 40));
  renderGallery();
}

function playShutterEffect() {
  dom.wrap.style.transition = 'none';
  dom.wrap.style.backgroundColor = '#ffffff';
  dom.video.style.opacity = '0';
  if (navigator.vibrate) navigator.vibrate(50);
  setTimeout(() => {
    dom.wrap.style.transition = 'background-color 0.3s ease-out';
    dom.video.style.transition = 'opacity 0.3s ease-out';
    dom.wrap.style.backgroundColor = '#000000';
    dom.video.style.opacity = '1';
  }, 50);
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
    const captures = loadCaptures();
    captures.unshift(canvas.toDataURL('image/jpeg', 0.95));
    saveCaptures(captures.slice(0, 40));
    renderGallery();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

dom.startBtn.addEventListener('click', startCamera);
dom.flipBtn.addEventListener('click', () => {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  startCamera();
});
dom.iosBtn.addEventListener('click', () => dom.iosCapture.click());
dom.iosCapture.addEventListener('change', (e) => {
  handleIOSPhoto(e.target.files?.[0]);
  e.target.value = '';
});
dom.clearBtn.addEventListener('click', () => {
  saveCaptures([]);
  renderGallery();
});
dom.saveAllBtn.addEventListener('click', saveAllToDevice);

dom.overlay.addEventListener('click', (event) => {
  const hit = pickFaceAtPoint(event.clientX, event.clientY);
  if (hit) {
    playShutterEffect();
    captureFace(hit);
  }
});

window.addEventListener('resize', resizeOverlay);
window.addEventListener('beforeunload', stopCamera);

renderGallery();
setStatus('Ready');
