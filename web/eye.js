/* eye.js — a procedural eyeball to replace the avatar's stock eye (a coarse front cap
 * only ~34° wide with a stretched iris strip: faceted, and empty behind once the eye
 * rotates far enough to expose the upper sclera).
 *
 *   const eye = createEye(radiusWorld, irisImage);   // irisImage: the avatar's unrolled iris strip
 *   bone.add(eye.group)                                // centre = eye-bone origin; orient with eye.aim()
 *   eye.aim(gazeAxisLocal, upAxisLocal, boneWorldScale)
 *   eye.setPupil(mm)                                   // pupil DIAMETER, mm — per frame, per eye
 *   eye.setCornea(visible)                             // hide while the lid is shut (dome would poke through)
 *
 * Globe: smooth sphere; iris / pupil / limbus / sclera are drawn in the fragment shader
 * from the polar angle about the gaze pole, so the pupil size is a uniform (dynamic,
 * per eye) and the iris pattern compresses as the pupil dilates. Cornea: a clear dome
 * over the iris (radius of curvature 0.75 R), drawn additively so only its glossy
 * highlights show.
 */
import * as THREE from 'three';

const IRIS_HALF_DEG  = 27;      // iris radius as an angle about the gaze pole (stylized avatar eye)
const IRIS_DIAM_MM   = 11.8;    // anatomical iris diameter → pupil mm maps to a fraction of it
const CORNEA_RC      = 0.75;    // cornea radius of curvature / globe radius (flatter than real 0.65
                                //   so the dome stays inside the lids: ~0.8 mm bulge here)
const SCLERA_SRGB    = [236, 232, 226];

// Crop the avatar's iris strip (rows: black pupil → iris → dark limbal ring → grey ramp →
// sclera) to just pupil-edge → end of limbal ring, so t = 0..1 spans the iris itself.
function irisStrip(image) {
  const c = document.createElement('canvas');
  c.width = image.width; c.height = image.height;
  const g = c.getContext('2d');
  g.drawImage(image, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  const lum = (y) => {
    let s = 0;
    for (let x = 0, i = y * c.width * 4; x < c.width; x++, i += 4) s += d[i] + d[i + 1] + d[i + 2];
    return s / (3 * c.width);
  };
  let top = 0;                                                  // first row past the black pupil
  while (top < c.height - 1 && lum(top) < 15) top++;
  let ring = Math.floor(c.height * 0.7);                        // darkest lower row = limbal ring
  for (let y = ring + 1; y < c.height; y++) if (lum(y) < lum(ring)) ring = y;
  let end = ring;                                               // last row of the ring
  while (end < c.height - 1 && lum(end + 1) < lum(ring) + 15) end++;
  const out = document.createElement('canvas');
  out.width = c.width; out.height = Math.max(1, end - top + 1);
  out.getContext('2d').drawImage(c, 0, top, c.width, out.height, 0, 0, c.width, out.height);
  const tex = new THREE.CanvasTexture(out);
  tex.flipY = false;                                            // canvas row 0 (pupil edge) → t = 0
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;                             // azimuth wraps around the iris
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 4;
  return tex;
}

export function createEye(radius, irisImage) {
  const group = new THREE.Group();          // eye-centred; +Z = gaze, +Y = eye up (set by aim)
  const thIris = THREE.MathUtils.degToRad(IRIS_HALF_DEG);
  const uniforms = {
    uThIris:  { value: thIris },
    uThPupil: { value: thIris * 4 / IRIS_DIAM_MM },
    uSclera:  { value: new THREE.Color().setRGB(...SCLERA_SRGB.map((v) => v / 255), THREE.SRGBColorSpace) },
  };

  // ── Globe ────────────────────────────────────────────────────────────────────
  const globeGeo = new THREE.SphereGeometry(radius, 96, 64);
  globeGeo.rotateX(Math.PI / 2);            // sphere pole (+Y, uv.y = 1) → +Z = gaze
  const globeMat = new THREE.MeshStandardMaterial({ map: irisStrip(irisImage), roughness: 0.45, metalness: 0 });
  globeMat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform float uThIris; uniform float uThPupil; uniform vec3 uSclera;`)
      .replace('#include <map_fragment>', `
        // polar angle about the gaze pole → pupil | iris (strip, compressed by the pupil) | sclera
        float th = (1.0 - vMapUv.y) * PI;
        vec3 iris = texture2D(map, vec2(vMapUv.x, clamp((th - uThPupil) / (uThIris - uThPupil), 0.0, 1.0))).rgb;
        vec3 col = mix(vec3(0.012), iris, smoothstep(uThPupil - 0.008, uThPupil + 0.008, th));
        col = mix(col, uSclera, smoothstep(uThIris - 0.006, uThIris + 0.012, th));
        col *= 1.0 - 0.45 * exp(-pow((th - uThIris) / 0.03, 2.0));                // soft limbal ring
        col = mix(col, col * vec3(0.93, 0.86, 0.84), smoothstep(uThIris + 0.25, 1.5, th));  // warmer periphery
        diffuseColor.rgb *= col;`);
  };
  const globe = new THREE.Mesh(globeGeo, globeMat);

  // ── Cornea: spherical cap meeting the globe at the limbus ─────────────────────
  const a  = radius * Math.sin(thIris);                     // limbus circle radius
  const zl = radius * Math.cos(thIris);                     // limbus plane along the gaze axis
  const rc = CORNEA_RC * radius;
  const zc = zl - Math.sqrt(rc * rc - a * a);               // cornea sphere centre on the axis
  const corneaGeo = new THREE.SphereGeometry(rc, 64, 24, 0, 2 * Math.PI, 0, Math.asin(a / rc));
  corneaGeo.rotateX(Math.PI / 2);
  corneaGeo.translate(0, 0, zc);
  // Black + additive: the dome adds only its specular highlights (wet catch-lights) on top
  // of the iris, so it never tints or hides what is under it.
  const corneaMat = new THREE.MeshPhysicalMaterial({
    color: 0x000000, roughness: 0.06, metalness: 0, ior: 1.376,
    clearcoat: 1, clearcoatRoughness: 0.04,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const cornea = new THREE.Mesh(corneaGeo, corneaMat);
  cornea.renderOrder = 1;
  group.add(globe, cornea);

  // Orient in the bone's local frame: +Z → gaze axis, +Y → eye up; undo the bone's world
  // scale so `radius` stays in world units.
  const _m = new THREE.Matrix4(), _x = new THREE.Vector3(), _y = new THREE.Vector3(), _z = new THREE.Vector3();
  function aim(gazeLocal, upLocal, boneWorldScale) {
    _z.copy(gazeLocal).normalize();
    _y.copy(upLocal).addScaledVector(_z, -upLocal.dot(_z)).normalize();
    _x.crossVectors(_y, _z);
    group.quaternion.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
    group.scale.setScalar(1 / (boneWorldScale || 1));
  }
  function setPupil(mm) {
    if (mm == null || !isFinite(mm)) return;
    uniforms.uThPupil.value = thIris * Math.min(0.8, Math.max(0.1, mm / IRIS_DIAM_MM));
  }
  function setCornea(visible) { cornea.visible = visible; }

  return { group, aim, setPupil, setCornea };
}
