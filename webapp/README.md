# DeepSonar 3D

Open a raw sensor `.npy` file and get four 3D views: raw input, cleaned input, object point cloud and 3D object.
The model runs in the browser, on the CPU of the device that opens the page. The server only hands over files.

    python3 -m http.server 8000 --directory webapp

- This computer: http://localhost:8000
- Phone or other computer on the same Wi-Fi: http://<this computer's IP>:8000

The first visit downloads the 139 MB model. Use a recent browser; low-memory phones may not be able to load it.
The back and hidden sides are predicted, not measured.

`pipeline.js`, `clean.js` and `clean_numerics.js` are the tested port of the Python pipeline. `model/` comes from
`data/adapointr.onnx`. Test the surface step with `node webapp/test/surface.test.mjs`.
