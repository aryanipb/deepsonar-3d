// Desktop shell: serves the web app to itself on a private local port and shows it. The model runs in this window.
const {app, BrowserWindow, shell} = require('electron');
const fs = require('fs'), http = require('http'), path = require('path');

const root = app.isPackaged ? path.join(process.resourcesPath, 'webapp') : path.join(__dirname, '..', 'webapp');
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm'};

const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]);
  const file = path.normalize(path.join(root, rel === '/' ? 'index.html' : rel));
  if (!file.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, {'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Content-Length': st.size});
    fs.createReadStream(file).pipe(res);
  });
});

app.whenReady().then(() => server.listen(0, '127.0.0.1', () => {
  const win = new BrowserWindow({width: 1280, height: 900, title: 'DeepSonar 3D', autoHideMenuBar: true});
  win.loadURL(`http://127.0.0.1:${server.address().port}/`);
  win.webContents.setWindowOpenHandler(({url}) => { shell.openExternal(url); return {action: 'deny'}; });
}));
app.on('window-all-closed', () => { server.close(); app.quit(); });
