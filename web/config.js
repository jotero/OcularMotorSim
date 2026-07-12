// Single source of truth for the simulation backend URL.
//
// Used as the API base when the app is NOT served by the backend itself — e.g. the
// static mirror at omlab.berkeley.edu/sim. When the app IS served from the backend's
// own origin (localhost during dev, or https://sim.oteromillan.com via the Cloudflare
// tunnel), same-origin is used automatically and this value is ignored.
//
// To move the backend (new tunnel/host), change this ONE line and redeploy.
window.OCULOMOTOR_BACKEND = 'https://sim.oteromillan.com';
