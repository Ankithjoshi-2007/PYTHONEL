// app.js - Web dashboard animation for 2D wind tunnel
// ---------------------------------------------------
// This script creates a canvas animation where the free‑stream wind direction is constant
// (horizontal flow from left to right) and only the wing orientation changes based on the
// user‑controlled Angle‑of‑Attack (α) slider. The wind speed is taken from the "True Airspeed"
// slider, but the direction never changes regardless of α. Particles are advected by the
// uniform flow and reflected when they intersect the wing geometry.

// ---------------------------------------------------
// Utility functions
const toRadians = deg => (deg * Math.PI) / 180;

// ---------------------------------------------------
// Global state
const canvas = document.getElementById('wind-tunnel-canvas');
const ctx = canvas.getContext('2d');
let width, height;

// UI elements
const aoaSlider = document.getElementById('slider-aoa');
const windSlider = document.getElementById('slider-wind');
const pauseBtn = document.getElementById('btn-pause-animation');
let paused = false;

// Animation parameters
const NUM_PARTICLES = 300;
const PARTICLE_SPEED_FACTOR = 0.2; // scales wind speed to pixel movement per frame
const WING_LENGTH = 200; // pixel length of the wing line (before scaling)
const WING_CHORD = 1; // placeholder, not used for line geometry
let particles = [];
let lastTimestamp = 0;

// ---------------------------------------------------
// Initialise canvas size (responsive)
function resizeCanvas() {
  const container = canvas.parentElement;
  width = container.clientWidth;
  height = container.clientHeight;
  canvas.width = width;
  canvas.height = height;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ---------------------------------------------------
// Particle object
class Particle {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  update(vx, vy) {
    this.x += vx;
    this.y += vy;
    // Wrap around when leaving the canvas
    if (this.x > width) this.x = 0;
    if (this.x < 0) this.x = width;
    if (this.y > height) this.y = 0;
    if (this.y < 0) this.y = height;
  }
  draw(ctx) {
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.fillRect(this.x, this.y, 2, 2);
  }
}

function initParticles() {
  particles = [];
  for (let i = 0; i < NUM_PARTICLES; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    particles.push(new Particle(x, y));
  }
}
initParticles();

// ---------------------------------------------------
// Wing geometry – represented as a line segment centered in the canvas
function getWingEndpoints(alphaDeg) {
  const cx = width / 2;
  const cy = height / 2;
  const halfLen = WING_LENGTH / 2;
  // Base line points (horizontal)
  const p1 = { x: -halfLen, y: 0 };
  const p2 = { x: halfLen, y: 0 };
  // Rotate points about origin by -alpha (wing rotates opposite to flow angle)
  const theta = -toRadians(alphaDeg);
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const rot = p => ({ x: p.x * cos - p.y * sin, y: p.x * sin + p.y * cos });
  const r1 = rot(p1);
  const r2 = rot(p2);
  // Translate to canvas centre
  return {
    x1: cx + r1.x,
    y1: cy + r1.y,
    x2: cx + r2.x,
    y2: cy + r2.y,
  };
}

// ---------------------------------------------------
// Simple collision detection – reflect particle when it crosses the wing line
function reflectIfCrossed(particle, wing) {
  // Vector from p1 to p2
  const dx = wing.x2 - wing.x1;
  const dy = wing.y2 - wing.y1;
  const lenSq = dx * dx + dy * dy;
  // Projection of particle onto the line segment
  const t = ((particle.x - wing.x1) * dx + (particle.y - wing.y1) * dy) / lenSq;
  if (t < 0 || t > 1) return; // outside segment
  // Closest point on the line
  const projX = wing.x1 + t * dx;
  const projY = wing.y1 + t * dy;
  const distSq = (particle.x - projX) ** 2 + (particle.y - projY) ** 2;
  const THRESH = 4; // pixel threshold squared
  if (distSq < THRESH) {
    // Normal vector of the wing (perpendicular to line)
    const nx = -dy;
    const ny = dx;
    const norm = Math.hypot(nx, ny);
    const nxU = nx / norm;
    const nyU = ny / norm;
    // Velocity vector of particle (uniform flow)
    // This function receives the uniform velocity components; we'll use globals below.
    const vx = particle.vx;
    const vy = particle.vy;
    // Reflect velocity: v' = v - 2 (v·n) n
    const dot = vx * nxU + vy * nyU;
    particle.vx = vx - 2 * dot * nxU;
    particle.vy = vy - 2 * dot * nyU;
    // Push particle slightly away from wing to avoid sticking
    particle.x = projX + nxU * 2;
    particle.y = projY + nyU * 2;
  }
}

// ---------------------------------------------------
// Main animation loop
function animate(timestamp) {
  if (paused) {
    requestAnimationFrame(animate);
    return;
  }
  const delta = timestamp - lastTimestamp;
  lastTimestamp = timestamp;

  // Clear canvas (dark background for premium look)
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, width, height);

  // Retrieve current UI values
  const alpha = parseFloat(aoaSlider.value); // degrees
  const windSpeed = parseFloat(windSlider.value); // m/s (or user unit)
  const vx = windSpeed * PARTICLE_SPEED_FACTOR; // pixels per frame (horizontal)
  const vy = 0; // wind direction fixed horizontally

  // Update and draw particles
  for (const p of particles) {
    // Store velocity for collision handling
    p.vx = vx;
    p.vy = vy;
    p.update(vx, vy);
    p.draw(ctx);
  }

  // Draw wing
  const wing = getWingEndpoints(alpha);
  ctx.strokeStyle = '#ffcc00';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(wing.x1, wing.y1);
  ctx.lineTo(wing.x2, wing.y2);
  ctx.stroke();

  // Collision handling – reflect particles that intersect wing
  for (const p of particles) {
    reflectIfCrossed(p, wing);
  }

  requestAnimationFrame(animate);
}

// ---------------------------------------------------
// UI listeners
aoaSlider.addEventListener('input', e => {
  document.getElementById('val-aoa').textContent = `${e.target.value}°`;
});
windSlider.addEventListener('input', e => {
  document.getElementById('val-wind').textContent = `${e.target.value} m/s`;
});
pauseBtn.addEventListener('click', () => {
  paused = !paused;
  pauseBtn.textContent = paused ? '▶ Play' : '⏸ Pause';
  if (!paused) requestAnimationFrame(animate);
});

// Start animation
requestAnimationFrame(animate);
