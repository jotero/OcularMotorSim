/* room.js — a furnished room as the 3D view's visual surround (the alternative to
 * the dot cloud). Real geometry, so head translation and vergence give true depth
 * parallax in the world and retina views.
 *
 *   const room = createRoom(unitsPerMetre);
 *   room.group                    // add to the scene at the rest eye-midpoint; rotate it for OKN
 *   room.setFront(zMetres)        // front-wall distance (push back for far targets)
 *   room.setHeadOffset(hd)        // head linear displacement [x,y,z] m (sim frame) → room shifts opposite
 *   room.hideBetween(camPos, eye) // hide furniture between a camera and the eye (world view); null → show all
 *
 * Built in the SUBJECT frame, metres, origin at the eye midpoint: x = right, y = up,
 * z = forward. The avatar's world flips x (avatar +X = subject LEFT), so every position
 * goes through P(). Assets: Poly Haven, CC0 — see assets/room/README.md.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const BASE = 'assets/room/';
const HALF_W = 2.4, FLOOR = -1.4, CEIL = 1.25, BACK = -1.8;   // metres from the eye
export const ROOM_MIN_FRONT = 4.0;                           // front wall (m); pushed back for far targets
const WALL_GAP = 0.01;                                       // furniture stands 1 cm off its wall (m)

// Facing → rotation about +Y in the avatar frame (the models face +Z as delivered).
const FACE = { fwd: 0, back: Math.PI, right: -Math.PI / 2, left: Math.PI / 2 };

// Furniture, subject frame. `anchor` = which point of the model's bounding box lands on
// (x, y, z): 'floor-back' (bottom of its back face — set x/z on a wall line), 'floor'
// (bottom centre), 'wall' (centre of its back face), 'ceiling' (top centre).
// front:true → z is an offset from the front wall (moves with it).
const ITEMS = [
  // left wall: sofa + picture above it, side table nearer the subject
  { name: 'Sofa_01',                  x: -HALF_W, y: FLOOR, z: 1.9, anchor: 'floor-back', face: 'right' },
  { name: 'hanging_picture_frame_01', x: -HALF_W, y: 0.25,  z: 1.9, anchor: 'wall',       face: 'right' },
  { name: 'side_table_01',            x: -HALF_W, y: FLOOR, z: 0.8, anchor: 'floor-back', face: 'right' },
  // right wall: bookshelf + picture, armchair angled into the room
  { name: 'wooden_bookshelf_worn',    x: HALF_W,  y: FLOOR, z: 2.5, anchor: 'floor-back', face: 'left' },
  { name: 'hanging_picture_frame_02', x: HALF_W,  y: 0.2,   z: 1.0, anchor: 'wall',       face: 'left' },
  { name: 'ArmChair_01',              x: 1.75,    y: FLOOR, z: 0.3, anchor: 'floor',      face: 'left', turn: -0.5 },
  // front wall: cabinet, clock, plant in the corner (window is procedural, below)
  { name: 'drawer_cabinet',           x: 1.5,     y: FLOOR, z: 0,     anchor: 'floor-back', face: 'back', front: true },
  { name: 'wall_clock',               x: 0.35,    y: 0.55,  z: 0,     anchor: 'wall',       face: 'back', front: true },
  { name: 'potted_plant_02',          x: -1.95,   y: FLOOR, z: -0.45, anchor: 'floor',      face: 'back', front: true },
  // ceiling
  { name: 'modern_ceiling_lamp_01',   x: 0,       y: CEIL,  z: 1.6, anchor: 'ceiling',    face: 'fwd' },
];
const WINDOW = { x: -0.85, y: 0.15, w: 1.3, h: 1.15 };       // on the front wall (m)

export function createRoom(unit) {
  const P = (x, y, z) => new THREE.Vector3(-x * unit, y * unit, z * unit);
  const group = new THREE.Group();     // sits at the eye; rotates with the scene (OKN)
  const world = new THREE.Group();     // shifted opposite the head's linear displacement
  const front = new THREE.Group();     // front wall + everything mounted on / against it
  group.add(world); world.add(front);
  group.visible = false;

  // ── Shell: floor, ceiling, four walls (one-sided, facing in, so an outside camera
  // sees straight through the near walls — a dollhouse view of the subject). ──────
  const texLoader = new THREE.TextureLoader();
  function surface(file, tileM, color) {
    const map = texLoader.load(BASE + 'textures/' + file);
    map.colorSpace = THREE.SRGBColorSpace;
    map.wrapS = map.wrapT = THREE.RepeatWrapping;
    map.anisotropy = 4;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
      new THREE.MeshStandardMaterial({ map, color, roughness: 0.95, metalness: 0 }));
    mesh.userData.tile = tileM;
    return mesh;
  }
  const size = (mesh, wM, hM) => {     // plane is 1×1 → scale to metres; keep texture tiles square
    mesh.scale.set(wM * unit, hM * unit, 1);
    mesh.material.map.repeat.set(wM / mesh.userData.tile, hM / mesh.userData.tile);
  };
  const WALL = 0xf1ebe1, WIDTH = 2 * HALF_W, HEIGHT = CEIL - FLOOR;
  const floor   = surface('laminate_floor_02_diff_1k.jpg', 1.6, 0xffffff);
  const ceiling = surface('painted_plaster_wall_diff_1k.jpg', 2.0, 0xfbfaf7);
  const wallL   = surface('painted_plaster_wall_diff_1k.jpg', 2.0, WALL);
  const wallR   = surface('painted_plaster_wall_diff_1k.jpg', 2.0, WALL);
  const wallB   = surface('painted_plaster_wall_diff_1k.jpg', 2.0, WALL);
  const wallF   = surface('painted_plaster_wall_diff_1k.jpg', 2.0, WALL);
  floor.rotation.x = -Math.PI / 2;  ceiling.rotation.x = Math.PI / 2;
  wallL.rotation.y = FACE.right;    wallR.rotation.y = FACE.left;   wallF.rotation.y = FACE.back;
  wallL.position.copy(P(-HALF_W, (FLOOR + CEIL) / 2, 0));
  wallR.position.copy(P(HALF_W, (FLOOR + CEIL) / 2, 0));
  wallB.position.copy(P(0, (FLOOR + CEIL) / 2, BACK));
  wallF.position.copy(P(0, (FLOOR + CEIL) / 2, 0));            // z set by the front group
  size(wallB, WIDTH, HEIGHT); size(wallF, WIDTH, HEIGHT);
  world.add(floor, ceiling, wallL, wallR, wallB); front.add(wallF);

  // ── Window (procedural): bright sky + distant hills behind a white frame. Unlit so it
  // reads as daylight. Sits 2 cm proud of the wall (clear of depth-buffer fighting). ──
  {
    const c = document.createElement('canvas'); c.width = 256; c.height = 224;
    const g = c.getContext('2d');
    const sky = g.createLinearGradient(0, 0, 0, c.height);
    sky.addColorStop(0, '#8fbfe8'); sky.addColorStop(0.65, '#dcebf7'); sky.addColorStop(1, '#eef5fb');
    g.fillStyle = sky; g.fillRect(0, 0, c.width, c.height);
    const hills = (y0, amp, color) => {
      g.fillStyle = color; g.beginPath(); g.moveTo(0, c.height);
      for (let x = 0; x <= c.width; x += 8) g.lineTo(x, y0 - amp * (0.6 * Math.sin(x / 37) + 0.4 * Math.sin(x / 13 + 1)));
      g.lineTo(c.width, c.height); g.fill();
    };
    hills(160, 18, '#a9bfa8'); hills(185, 12, '#7f9f7c');
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const W = WINDOW;
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(W.w * unit, W.h * unit),
      new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }));
    pane.rotation.y = FACE.back; pane.position.copy(P(W.x, W.y, -0.02));
    const frameMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0 });
    const bar = (w, h, x, y) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w * unit, h * unit, 0.05 * unit), frameMat);
      m.position.copy(P(x, y, -0.03)); return m;
    };
    const t = 0.06;
    front.add(pane,
      bar(W.w + 2 * t, t, W.x, W.y + W.h / 2 + t / 2), bar(W.w + 2 * t, t, W.x, W.y - W.h / 2 - t / 2),
      bar(t, W.h, W.x - W.w / 2 - t / 2, W.y),         bar(t, W.h, W.x + W.w / 2 + t / 2, W.y),
      bar(0.04, W.h, W.x, W.y),                        bar(W.w, 0.04, W.x, W.y));
  }

  // ── Furniture (glTF, loaded lazily — appears as each file arrives). ─────────────
  const pivots = [];
  const loader = new GLTFLoader();
  const _box = new THREE.Box3();
  for (const it of ITEMS) {
    loader.load(`${BASE}models/${it.name}/${it.name}.gltf`, (gltf) => {
      const model = gltf.scene, pivot = new THREE.Group();
      // Drop glass sheets (picture frames, clock): with the extra maps stripped they render
      // as an opaque-looking grey pane, and blending them over the layer ~1 mm behind flickers.
      model.traverse((o) => { if (o.isMesh && /_glass$/.test(o.material.name || '')) o.visible = false; });
      pivot.add(model);
      model.scale.setScalar(unit);
      model.updateMatrixWorld(true);
      _box.setFromObject(model);
      const c = _box.getCenter(new THREE.Vector3());
      const back = _box.min.z - WALL_GAP * unit;    // stand off the wall: coplanar faces flicker
      const a = it.anchor === 'floor-back' ? new THREE.Vector3(c.x, _box.min.y, back)
              : it.anchor === 'floor'      ? new THREE.Vector3(c.x, _box.min.y, c.z)
              : it.anchor === 'wall'       ? new THREE.Vector3(c.x, c.y, back)
              :                              new THREE.Vector3(c.x, _box.max.y, c.z);   // ceiling
      model.position.sub(a);
      pivot.rotation.y = FACE[it.face] + (it.turn || 0);
      pivot.position.copy(P(it.x, it.y, it.z));
      (it.front ? front : world).add(pivot);
      pivots.push(pivot);
    }, undefined, (err) => console.warn('room: failed to load', it.name, err));
  }

  let frontM = ROOM_MIN_FRONT;
  function setFront(zM) {
    frontM = Math.max(ROOM_MIN_FRONT, zM || 0);
    const L = frontM - BACK, zMid = (frontM + BACK) / 2;
    front.position.z = frontM * unit;
    floor.position.copy(P(0, FLOOR, zMid));   size(floor, WIDTH, L);
    ceiling.position.copy(P(0, CEIL, zMid));  size(ceiling, WIDTH, L);
    wallL.position.z = wallR.position.z = zMid * unit;
    size(wallL, L, HEIGHT); size(wallR, L, HEIGHT);
  }
  setFront(ROOM_MIN_FRONT);

  // Sim-frame displacement (m) → avatar frame, opposite the head (same as the dot cloud).
  function setHeadOffset(hd) {
    if (hd) world.position.set(hd[0] * unit, -hd[1] * unit, -hd[2] * unit);
    else    world.position.set(0, 0, 0);
  }

  // World view: hide any piece of furniture lying on the camera's side of the eye, so it
  // can't block the view of the head (walls already vanish from outside: one-sided).
  const _ctr = new THREE.Vector3(), _toCam = new THREE.Vector3();
  function hideBetween(camPos, eyePos) {
    for (const p of pivots) {
      if (!camPos) { p.visible = true; continue; }
      _box.setFromObject(p).getCenter(_ctr);
      _toCam.copy(camPos).sub(eyePos);
      p.visible = _ctr.sub(eyePos).dot(_toCam) <= 0.15 * _toCam.lengthSq();
    }
  }

  return { group, setFront, setHeadOffset, hideBetween };
}
