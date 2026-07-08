# Neural Network Proof Lab

A static, browser-only webapp for four neural-network proof demos:

1. Activations exist for a reason.
2. Depth without nonlinearity is a lie.
3. Embeddings learn similarity from next-token prediction.
4. Memorization vs generalization, and data closes the gap.

## Deploy on Netlify

### Option A: Drag and drop

1. Go to Netlify.
2. Open **Sites**.
3. Drag the entire `neural-proof-webapp` folder into Netlify's deploy area.
4. Netlify will host it and give you a link.

### Option B: Upload the ZIP

Upload `neural-proof-webapp.zip` to Netlify if you prefer deploying a compressed copy.

## Files

- `index.html` — app structure
- `style.css` — visual design
- `app.js` — all data generation, model training, plotting, and proof logic
- `README.md` — deployment notes

There is no build step, no package install, and no backend.
