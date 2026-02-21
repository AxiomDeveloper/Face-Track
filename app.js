import { fileSystem } from './data.js';

const DOM = {
    output: document.getElementById('output'),
    path: document.getElementById('current-path'),
    builtCmd: document.getElementById('built-cmd'),
    buttons: document.getElementById('context-buttons')
};

let pwd = "/";
let activeCmd = []; // Stores the command being built, e.g., ['mv', 'file.txt', 'target_dir']

function print(text, className = 'log-sys') {
    const div = document.createElement('div');
    div.className = className;
    div.textContent = text;
    DOM.output.appendChild(div);
    DOM.output.scrollTop = DOM.output.scrollHeight;
}

// Gets files/dirs in current path
function getContents(typeFilter = null) {
    const node = fileSystem[pwd];
    if (!node || node.type !== 'dir') return [];
    if (!typeFilter) return node.contents;
    
    // Filter by type (e.g., only show directories for 'cd')
    return node.contents.filter(item => {
        const itemPath = pwd === '/' ? `/${item}` : `${pwd}/${item}`;
        return fileSystem[itemPath].type === typeFilter;
    });
}

function updateUI() {
    DOM.path.textContent = pwd;
    DOM.builtCmd.textContent = activeCmd.join(' ');
    renderButtons();
}

function handleButtonPress(value, action = 'append') {
    if (action === 'append') {
        activeCmd.push(value);
    } else if (action === 'cancel') {
        activeCmd = [];
    } else if (action === 'execute') {
        executeCommand();
    }
    updateUI();
}

function renderButtons() {
    DOM.buttons.innerHTML = '';
    
    // STATE 0: No command selected. Show Base Commands.
    if (activeCmd.length === 0) {
        createBtn('ls (List)', 'ls');
        createBtn('cd (Open Dir)', 'cd');
        createBtn('cat (Read File)', 'cat');
        createBtn('mv (Move File)', 'mv');
        return;
    }

    const baseCmd = activeCmd[0];

    // STATE 1: 'ls' is selected. Ready to execute.
    if (baseCmd === 'ls') {
        createBtn('EXECUTE', '', 'execute', 'btn-exec');
        createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        return;
    }

    // STATE 2: 'cd' is selected. Show available directories.
    if (baseCmd === 'cd') {
        if (activeCmd.length === 1) {
            if (pwd !== '/') createBtn('.. (Go Back)', '..');
            getContents('dir').forEach(dir => createBtn(dir, dir));
            createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        } else {
            createBtn('EXECUTE', '', 'execute', 'btn-exec');
            createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        }
        return;
    }

    // STATE 3: 'cat' is selected. Show available files.
    if (baseCmd === 'cat') {
        if (activeCmd.length === 1) {
            getContents('file').forEach(file => createBtn(file, file));
            createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        } else {
            createBtn('EXECUTE', '', 'execute', 'btn-exec');
            createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        }
        return;
    }

    // STATE 4: 'mv' is selected. Needs File, then Needs Directory.
    if (baseCmd === 'mv') {
        if (activeCmd.length === 1) {
            // Step 1: Pick file to move
            getContents('file').forEach(file => createBtn(file, file));
            createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        } else if (activeCmd.length === 2) {
            // Step 2: Pick destination
            const dirs = ['/secure_nodes/financial_crimes', '/secure_nodes/surveillance', '/secure_nodes/profiles'];
            dirs.forEach(d => createBtn(`TO: ${d.split('/').pop()}`, d));
            createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        } else {
            createBtn('EXECUTE', '', 'execute', 'btn-exec');
            createBtn('CANCEL', '', 'cancel', 'btn-cancel');
        }
        return;
    }
}

function createBtn(label, value, action = 'append', cssClass = '') {
    const btn = document.createElement('button');
    btn.textContent = label;
    if (cssClass) btn.className = cssClass;
    btn.onclick = () => handleButtonPress(value, action);
    DOM.buttons.appendChild(btn);
}

// --- COMMAND EXECUTION LOGIC ---
function executeCommand() {
    const cmdStr = activeCmd.join(' ');
    print(`root@mi5:${pwd}$ ${cmdStr}`, 'log-cmd');
    
    const cmd = activeCmd[0];
    const arg1 = activeCmd[1];
    const arg2 = activeCmd[2];

    activeCmd = []; // Reset builder immediately

    if (cmd === 'ls') {
        const contents = getContents();
        if (contents.length === 0) print('(empty directory)', 'log-sys');
        else print(contents.join('   '), 'log-data');
    } 
    else if (cmd === 'cd') {
        if (arg1 === '..') {
            const parts = pwd.split('/').filter(Boolean);
            parts.pop();
            pwd = parts.length === 0 ? '/' : '/' + parts.join('/');
        } else {
            pwd = pwd === '/' ? `/${arg1}` : `${pwd}/${arg1}`;
        }
    }
    else if (cmd === 'cat') {
        const targetPath = pwd === '/' ? `/${arg1}` : `${pwd}/${arg1}`;
        print(fileSystem[targetPath].content, 'log-data');
    }
    else if (cmd === 'mv') {
        const srcPath = pwd === '/' ? `/${arg1}` : `${pwd}/${arg1}`;
        const destPath = arg2;
        const filename = arg1;

        // Move logic
        fileSystem[`${destPath}/${filename}`] = fileSystem[srcPath];
        delete fileSystem[srcPath];

        // Update arrays
        fileSystem[pwd].contents = fileSystem[pwd].contents.filter(f => f !== filename);
        fileSystem[destPath].contents.push(filename);

        print(`SECURE TRANSFER: ${filename} routed to ${destPath}`, 'log-success');
    }
}

// Boot Sequence
print("INITIALIZING MI5 SECURE KERNEL...", "log-sys");
setTimeout(() => {
    print("Connection established. Awaiting input.", "log-success");
    updateUI();
}, 600);