/* avatar.js — shared 3D head + playback viewer (three.js module).
 *
 * Used by both the simulator page (index.html) and the gallery (gallery.html).
 * Expects these elements to exist in the page (same markup block on both):
 *   #avatar-canvas, #avatar-loading, #play-btn, #scrubber, #time-display,
 *   .avatar-wrap, .avatar-labels span
 *
 * Public API (on window):
 *   loadEyeTrajectory(traj)   — load a downsampled per-eye trajectory + play UI
 *   _avatarTogglePlay()       — play / pause
 *   _avatarOnScrub(frame)     — seek to a frame
 * Calls window.setPlotTime(seconds) each frame to sync the plot time-cursor.
 *
 * Requires an importmap defining "three" + "three/addons/" in the host page.
 */
import * as THREE     from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── Renderer + scene ──────────────────────────────────────────────────────────
const canvas   = document.getElementById('avatar-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.autoClear = false;   // we clear manually between viewports

const W = () => canvas.clientWidth  || 820;
const H = () => canvas.clientHeight || 420;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef0f4);

// Two cameras — world view (left) and head-fixed view (right)
const worldCam = new THREE.PerspectiveCamera(28, W() / 2 / H(), 0.001, 10000);
const headCam  = new THREE.PerspectiveCamera(28, W() / 2 / H(), 0.001, 10000);
// Layer 1 = world-only props (target sphere, gaze rays). World view sees layers
// 0+1; head-fixed close-up sees only layer 0 (no clutter over the eyeballs).
worldCam.layers.enable(1);

function resize() {
  renderer.setSize(W(), H(), false);
  // Camera aspect ratios are set per-frame in renderViewports() based on _headMoves
}
new ResizeObserver(resize).observe(canvas);
resize();

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(1, 3, 2); scene.add(key);
scene.add(new THREE.DirectionalLight(0x8899ff, 0.4)).position.set(-2, 1, 1);

// ── Avatar ────────────────────────────────────────────────────────────────────
let leftEyeBone  = null;
let rightEyeBone = null;
let headBone     = null;
let restL = null, restR = null, restHead = null;
let coverMeshL   = null, coverMeshR = null;
let faceMesh     = null;   // skinned mesh carrying the ARKit morph targets (eyelids)

// Target sphere + gaze rays. Everything in WORLD space; the eye anchor is the
// eye bone's getWorldPosition() (the canonical, already-correct world position —
// the same one the head-fixed camera uses).
let targetSphere = null, gazeRayL = null, gazeRayR = null;
let _restEyeMid  = null;            // world eye-mid at rest (for the world-fixed target)
let _gazeAxisL   = null, _gazeAxisR = null;  // eye-local axis that points along gaze
let _modelUnit   = 1;               // world units per metre (from eye separation)
let _hasTarget   = false, _showWorld = false;
// World-camera view presets (switchable via keys d/t/l/r — temporary debug aid).
let _camRefEye = null, _camRefSize = 1, _camNear = 0.01, _camFar = 100;

function setWorldView(mode) {
  if (!_camRefEye) return;
  const e = _camRefEye, s = _camRefSize, u = _modelUnit;
  worldCam.fov = 36;
  const fwd = 0.5 * u;   // aim at the eye→target midpoint (target ≈ 1 u in front)
  let pos, look;
  if (mode === 'top') {
    pos  = new THREE.Vector3(e.x, e.y + s * 1.9, e.z - s * 0.15);
    look = new THREE.Vector3(e.x, e.y, e.z + fwd);
  } else if (mode === 'left') {
    pos  = new THREE.Vector3(e.x - s * 1.8, e.y + s * 0.25, e.z + fwd);
    look = new THREE.Vector3(e.x, e.y, e.z + fwd);
  } else if (mode === 'right') {
    pos  = new THREE.Vector3(e.x + s * 1.8, e.y + s * 0.25, e.z + fwd);
    look = new THREE.Vector3(e.x, e.y, e.z + fwd);
  } else {  // default: behind + above
    pos  = new THREE.Vector3(e.x + s * 0.35, e.y + s * 0.65, e.z - s * 1.15);
    look = new THREE.Vector3(e.x, e.y - s * 0.05, e.z + fwd);
  }
  worldCam.position.copy(pos);
  worldCam.lookAt(look);
  worldCam.near = _camNear; worldCam.far = Math.max(_camFar, s * 60);
  worldCam.updateProjectionMatrix();
}

const AVATAR_PATH = 'avatar/avatar.glb';

new GLTFLoader().load(AVATAR_PATH, (gltf) => {
  const model = gltf.scene;
  scene.add(model);

  model.traverse(obj => {
    if (obj.name === 'LeftEye'  || obj.name === 'LeftEye_08')  leftEyeBone  = obj;
    if (obj.name === 'RightEye' || obj.name === 'RightEye_09') rightEyeBone = obj;
    // First skinned/standard mesh that carries ARKit blendshapes = the face.
    if (obj.isMesh && obj.morphTargetDictionary && !faceMesh) faceMesh = obj;
  });
  console.log('Face morph targets:', faceMesh ? Object.keys(faceMesh.morphTargetDictionary).length : 0);

  // Rotate the whole model for head/body movement — avoids neck artifacts
  headBone = model;

  if (leftEyeBone)  restL    = leftEyeBone.rotation.clone();
  if (rightEyeBone) restR    = rightEyeBone.rotation.clone();
  restHead = model.rotation.clone();
  console.log('Bones found — leftEye:', !!leftEyeBone, 'rightEye:', !!rightEyeBone);
  console.log('restHead:', restHead);

  // Use actual eye bone world positions for precise camera targeting
  const box  = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const near = size.y * 0.001, far = size.y * 20;

  const posL = new THREE.Vector3(), posR = new THREE.Vector3();
  leftEyeBone.getWorldPosition(posL);
  rightEyeBone.getWorldPosition(posR);
  const eyeMid = new THREE.Vector3().addVectors(posL, posR).multiplyScalar(0.5);

  // Head-fixed (right): close-up, eye midpoint target
  headCam.position.set(eyeMid.x, eyeMid.y, eyeMid.z + size.y * 0.18);
  headCam.lookAt(eyeMid);
  headCam.near = near; headCam.far = far; headCam.updateProjectionMatrix();

  // ── Target + gaze-ray anchor ──────────────────────────────────────────────
  // The skinned head renders WITHOUT the faceMesh node offset, so the rendered
  // eye = the bone world position mapped into faceMesh-local space (offset
  // removed). Confirmed on-screen: this point lands exactly on the eyes.
  _restEyeMid = faceMesh.worldToLocal(eyeMid.clone());
  // World units per metre, referenced to the SIM's IPD (SensoryParams.ipd = 0.064 m)
  // so the avatar's eye separation matches the IPD the sim used for the per-eye
  // angles — i.e. the two gaze rays converge exactly on the target.
  _modelUnit  = posL.distanceTo(posR) / 0.064;

  model.updateMatrixWorld(true);
  const qL0 = leftEyeBone.getWorldQuaternion(new THREE.Quaternion());
  const qR0 = rightEyeBone.getWorldQuaternion(new THREE.Quaternion());
  _gazeAxisL = new THREE.Vector3(0, 0, 1).applyQuaternion(qL0.clone().invert());
  _gazeAxisR = new THREE.Vector3(0, 0, 1).applyQuaternion(qR0.clone().invert());

  // Normal opaque material so the head correctly occludes the props (a target
  // in front of the face is hidden by the head from the behind camera).
  // frustumCulled:false because per-frame repositioning + dual-camera rendering
  // otherwise wrongly culls them.
  const overlay = (color) => new THREE.MeshBasicMaterial({ color, toneMapped: false });
  // Layer 0 (proven to render in the world view); hidden from the head-fixed
  // close-up manually in renderViewports. (Layer 1 did not render reliably here.)
  const prop = (mesh) => { mesh.frustumCulled = false; mesh.visible = false;
                           scene.add(mesh); return mesh; };

  targetSphere = prop(new THREE.Mesh(new THREE.SphereGeometry(0.022 * _modelUnit, 20, 14), overlay(0xe23b3b)));
  gazeRayL = prop(new THREE.Mesh(new THREE.CylinderGeometry(0.0022 * _modelUnit, 0.0022 * _modelUnit, 1, 8), overlay(0x2166ac)));   // left  — blue (matches plots)
  gazeRayR = prop(new THREE.Mesh(new THREE.CylinderGeometry(0.0022 * _modelUnit, 0.0022 * _modelUnit, 1, 8), overlay(0xd6604d)));   // right — red  (matches plots)


  // World (left) camera: stable side-three-quarter framing of head + target,
  // set ONCE here (after the model loads) so it never races the async load.
  // World camera: store reference frame, then apply the default view. Switch
  // views with d (default) / t (top) / l (left) / r (right).
  _camRefEye = eyeMid.clone(); _camRefSize = size.y; _camNear = near; _camFar = far;
  setWorldView('default');

  // Build procedural eye-cover patches — black disc parented to the model
  // root (acts as head bone) so they stay head-fixed: the eyeball rotates
  // beneath the disc while the disc stays put on the face.  Visibility is
  // toggled per frame from _traj.cover_L / _traj.cover_R.
  //
  // Material choices that matter:
  //   MeshBasicMaterial   — no lighting required, guaranteed solid black
  //   DoubleSide          — visible from either side (model rotates in world view)
  //   depthTest:false     — never occluded by face mesh
  //   renderOrder: 999    — drawn last so it lands on top of everything
  //   transparent:false   — avoid three.js transparent-sort quirks
  const coverGeo = new THREE.CircleGeometry(size.y * 0.028, 48);
  const coverMat = new THREE.MeshBasicMaterial({
    color: 0x000000,
    side: THREE.DoubleSide,
    depthTest:  false,
    depthWrite: false,
    transparent: false,
  });

  // Make sure all matrices are current before computing offsets.
  model.updateMatrixWorld(true);

  // Forward direction (toward camera) in MODEL-local coords.  The camera
  // sits at +Z relative to the eye midpoint in world space; the face is
  // oriented such that this is "out of the head" — exactly what we want.
  const fwdLocalModel = new THREE.Vector3(0, 0, 1)
      .transformDirection(new THREE.Matrix4().copy(model.matrixWorld).invert());

  // Build a temporary "cover anchor" in model-local space, just in front
  // of each eye, then reparent to the model root.  Parenting to model
  // (not the eye bone) keeps the cover head-fixed: the eye rotates
  // beneath the cover, the cover stays put on the face.
  const fwdOffset = size.y * 0.05;

  // posL / posR are world positions of the eye bones — convert to model-local.
  const eyeL_local = model.worldToLocal(posL.clone());
  const eyeR_local = model.worldToLocal(posR.clone());

  coverMeshL = new THREE.Mesh(coverGeo, coverMat);
  coverMeshR = new THREE.Mesh(coverGeo, coverMat.clone());
  coverMeshL.position.copy(eyeL_local).addScaledVector(fwdLocalModel, fwdOffset);
  coverMeshR.position.copy(eyeR_local).addScaledVector(fwdLocalModel, fwdOffset);
  coverMeshL.renderOrder = 999;
  coverMeshR.renderOrder = 999;
  coverMeshL.visible = false;
  coverMeshR.visible = false;
  model.add(coverMeshL);
  model.add(coverMeshR);

  // Orient discs to face the camera.  At load time the model is at its
  // rest pose, so the head-fixed view sees the discs head-on.  In world
  // view the model rotates and the cover rotates with it — DoubleSide
  // material keeps it visible from either side.
  coverMeshL.lookAt(headCam.position);
  coverMeshR.lookAt(headCam.position);

  console.log('Eye-cover meshes created, parented to avatar root, hidden by default.');

  document.getElementById('avatar-loading').style.display = 'none';

}, undefined, err => {
  console.error('Avatar load error:', err);
  document.getElementById('avatar-loading').textContent = 'Avatar failed to load';
});

// ── Eyelids (ARKit blendshapes) ────────────────────────────────────────────────
function setMorph(name, value) {
  if (!faceMesh) return;
  const i = faceMesh.morphTargetDictionary[name];
  if (i !== undefined) faceMesh.morphTargetInfluences[i] = value;
}

// Spontaneous blink: a short 0->1->0 close every few seconds (real-time clock,
// so the avatar looks alive even when paused). Advanced from animate(ts).
let _blink = 0;
let _nextBlinkTs = 0;
function updateBlink(ts) {
  if (_nextBlinkTs === 0) { _nextBlinkTs = ts + 2000 + Math.random() * 3500; return; }
  const since = ts - _nextBlinkTs;      // >= 0 once the blink has started
  const DUR = 150;                      // ms
  if (since >= 0 && since <= DUR) {
    const ph = since / DUR;
    _blink = ph < 0.5 ? ph * 2 : (1 - ph) * 2;   // triangle 0->1->0
  } else if (since > DUR) {
    _blink = 0;
    _nextBlinkTs = ts + 2000 + Math.random() * 3500;
  }
}

// ── Bone application ──────────────────────────────────────────────────────────
const DEG = Math.PI / 180;
const _UP = new THREE.Vector3(0, 1, 0);

// Map a sim target (metres, world Cartesian: x=right, y=up, z=forward) to a world
// offset from the eye. TRUE distance (no compression) so the two per-eye gaze
// rays actually converge on the sphere — compressing it would move the sphere off
// the convergence point. Sign on x matches the rendered eye (rightward → world −X).
function targetWorld(p) {
  return new THREE.Vector3(-p[0], p[1], p[2]).multiplyScalar(_modelUnit).add(_restEyeMid);
}

// One gaze-ray cylinder: from the eye's live world position along its gaze axis,
// reaching the target. Eye world pos + orientation come straight from the bone.
const _rO = new THREE.Vector3(), _rQ = new THREE.Quaternion();
function updateRay(cyl, bone, axis, show) {
  if (!cyl || !bone || !axis) return;
  if (!show) { cyl.visible = false; return; }
  bone.getWorldPosition(_rO);
  faceMesh.worldToLocal(_rO);   // remove the faceMesh node offset → rendered eye position
  bone.getWorldQuaternion(_rQ);
  const dir = axis.clone().applyQuaternion(_rQ).normalize();
  const len = targetSphere.visible ? Math.max(0.02, targetSphere.position.distanceTo(_rO))
                                   : TARGET_VIZ_DIST * _modelUnit;
  cyl.position.copy(_rO).addScaledVector(dir, len / 2);
  cyl.quaternion.setFromUnitVectors(_UP, dir);
  cyl.scale.set(1, len, 1);
  cyl.visible = true;
}

// eye_pos in simulation = head-fixed plant state [yaw, pitch, roll] deg
// head_pos              = integrated head velocity [yaw, pitch, roll] deg
function applyFrame(fi) {
  if (!leftEyeBone || !rightEyeBone || !_traj) return;
  const L = _traj.left[fi];
  const R = _traj.right[fi];

  leftEyeBone.rotation.set(
    restL.x - L[1] * DEG,   // pitch
    restL.y - L[0] * DEG,   // yaw
    restL.z
  );
  rightEyeBone.rotation.set(
    restR.x - R[1] * DEG,
    restR.y - R[0] * DEG,
    restR.z
  );

  // Head rotation for world view (applied before left render, removed before right)
  if (headBone && _traj.head) {
    const Hd = _traj.head[fi];
    headBone.rotation.set(
      restHead.x - Hd[1] * DEG,   // pitch
      restHead.y - Hd[0] * DEG,   // yaw
      restHead.z
    );
  }

  // Cover patches
  if (coverMeshL) coverMeshL.visible = !!(_traj.cover_L && _traj.cover_L[fi]);
  if (coverMeshR) coverMeshR.visible = !!(_traj.cover_R && _traj.cover_R[fi]);

  // Eyelids: spontaneous blink + upper lid follows vertical gaze (downgaze
  // lowers the lid via eyeBlink; upgaze retracts it via eyeWide). L/R = [yaw,
  // pitch, roll] deg; pitch > 0 = up.
  if (faceMesh) {
    const downL = Math.max(0, -L[1]) / 70, upL = Math.max(0, L[1]) / 45;
    const downR = Math.max(0, -R[1]) / 70, upR = Math.max(0, R[1]) / 45;
    setMorph('eyeBlinkLeft',  Math.min(1, Math.max(_blink, downL * 0.4)));
    setMorph('eyeBlinkRight', Math.min(1, Math.max(_blink, downR * 0.4)));
    setMorph('eyeWideLeft',  Math.min(0.5, upL) * (1 - _blink));
    setMorph('eyeWideRight', Math.min(0.5, upR) * (1 - _blink));
  }

  // Target sphere (world-fixed) + gaze rays (from the live eye bones).
  if (targetSphere && _restEyeMid) {
    const present = _traj.target && _traj.target[fi] &&
                    (!_traj.target_present || !!_traj.target_present[fi]);
    targetSphere.visible = !!present;
    if (present) targetSphere.position.copy(targetWorld(_traj.target[fi]));
    updateRay(gazeRayL, leftEyeBone,  _gazeAxisL, targetSphere.visible);
    updateRay(gazeRayR, rightEyeBone, _gazeAxisR, targetSphere.visible);
  }
}

// ── Playback ──────────────────────────────────────────────────────────────────
let _traj      = null;
let _frame     = 0;
let _playing   = false;
let _lastRafTs = null;
let _headMoves = false;

function updateTimeDisplay(fi) {
  if (!_traj) return;
  document.getElementById('time-display').textContent =
    `${(fi / _traj.fps).toFixed(1)} / ${_traj.duration_s.toFixed(1)} s`;
  // Sync the plot time-cursor to the current playback/scrub time.
  if (window.setPlotTime) window.setPlotTime(fi / _traj.fps);
}

window.loadEyeTrajectory = function(traj) {
  _traj    = traj;
  _frame   = 0;
  _playing = false;

  // Detect meaningful head movement (any axis > 1 deg peak displacement)
  _headMoves = false;
  if (traj.head) {
    const maxDisp = Math.max(
      ...traj.head.map(h => Math.sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]))
    );
    _headMoves = maxDisp > 1.0;
  }

  // One-time diagnostic: if a trajectory has any cover_L/R frames > 0,
  // log it so we can see in the console that the data path is intact.
  // (Data flows: stimuli.build_visual_flags → simulate → server _build_traj
  //  → cover_L/cover_R int arrays → window.loadEyeTrajectory.)
  if (traj.cover_L || traj.cover_R) {
    const sumL = (traj.cover_L || []).reduce((a, b) => a + b, 0);
    const sumR = (traj.cover_R || []).reduce((a, b) => a + b, 0);
    console.log(`Cover data: L=${sumL} frames covered, R=${sumR} frames covered.`);
  }

  // A foveal target makes the world view meaningful even without head movement.
  _hasTarget = !!traj.target && (!traj.target_present || traj.target_present.some(v => v));
  _showWorld = _headMoves || _hasTarget;

  // Show/hide world-view label + divider
  const labels = document.querySelectorAll('.avatar-labels span');
  if (labels[0]) labels[0].style.display = _showWorld ? '' : 'none';
  const wrap = document.querySelector('.avatar-wrap');
  if (wrap) wrap.classList.toggle('single-view', !_showWorld);

  document.getElementById('play-btn').textContent = '▶';
  document.getElementById('scrubber').max         = traj.n_frames - 1;
  document.getElementById('scrubber').value       = 0;
  updateTimeDisplay(0);
  applyFrame(0);
};

window._avatarTogglePlay = function() {
  if (!_traj) return;
  _playing = !_playing;
  if (_playing) {
    if (_frame >= _traj.n_frames - 1) _frame = 0;
    _lastRafTs = null;
    document.getElementById('play-btn').textContent = '⏸';
  } else {
    document.getElementById('play-btn').textContent = '▶';
  }
};

window._avatarOnScrub = function(val) {
  if (!_traj) return;
  _frame = val; _lastRafTs = null;
  applyFrame(val);
  updateTimeDisplay(val);
};

// ── Render loop (scissor split) ───────────────────────────────────────────────
function renderViewports() {
  const w = W(), h = H();
  const fi = Math.min(Math.floor(_frame), _traj ? _traj.n_frames - 1 : 0);

  renderer.clear();

  if (_showWorld) {
    // Split: left = world view, right = head-fixed view
    const hw = Math.floor(w / 2);
    const a  = hw / h;
    worldCam.aspect = a; worldCam.updateProjectionMatrix();
    headCam.fov = 28; headCam.aspect = a; headCam.updateProjectionMatrix();

    // Left — world view: apply head rotation
    applyFrame(fi);
    renderer.setViewport(0, 0, hw, h);
    renderer.setScissor(0, 0, hw, h);
    renderer.setScissorTest(true);
    renderer.render(scene, worldCam);

    // Right — head-fixed view: head bone at rest, eyes unchanged. Hide the
    // world-only props (target + rays) so they don't clutter the eyeball close-up.
    if (headBone && restHead) headBone.rotation.copy(restHead);
    const _pv = [targetSphere, gazeRayL, gazeRayR].map(m => m && m.visible);
    [targetSphere, gazeRayL, gazeRayR].forEach(m => { if (m) m.visible = false; });
    renderer.setViewport(hw, 0, w - hw, h);
    renderer.setScissor(hw, 0, w - hw, h);
    renderer.render(scene, headCam);
    [targetSphere, gazeRayL, gazeRayR].forEach((m, i) => { if (m) m.visible = _pv[i]; });
  } else {
    // No head movement — single full-width head-fixed view, zoomed in on eyes
    headCam.fov = 14; headCam.aspect = w / h; headCam.updateProjectionMatrix();
    applyFrame(fi);
    if (headBone && restHead) headBone.rotation.copy(restHead);
    renderer.setViewport(0, 0, w, h);
    renderer.setScissor(0, 0, w, h);
    renderer.setScissorTest(true);
    renderer.render(scene, headCam);
  }
}

function animate(ts) {
  requestAnimationFrame(animate);
  updateBlink(ts);   // spontaneous blink (eyelids), independent of playback

  if (_playing && _traj) {
    if (_lastRafTs !== null) {
      _frame += (ts - _lastRafTs) / 1000 * _traj.fps;
      if (_frame >= _traj.n_frames) {
        _frame = _traj.n_frames - 1; _playing = false;
        document.getElementById('play-btn').textContent = '▶';
      }
      const fi = Math.min(Math.floor(_frame), _traj.n_frames - 1);
      applyFrame(fi);
      document.getElementById('scrubber').value = fi;
      updateTimeDisplay(fi);
    }
    _lastRafTs = ts;
  }

  renderViewports();
}
requestAnimationFrame(animate);

// Temporary world-camera view controls: d=default, t=top, l=left, r=right.
// Ignored while typing in a form field.
window.addEventListener('keydown', (e) => {
  if (e.target && /^(input|textarea|select)$/i.test(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if      (k === 'd') setWorldView('default');
  else if (k === 't') setWorldView('top');
  else if (k === 'l') setWorldView('left');
  else if (k === 'r') setWorldView('right');
});
