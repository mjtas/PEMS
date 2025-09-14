const express = require('express');
const cors = require('cors');
const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');
const wss = new WebSocketServer({ port: 3001 });

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // Serve static files

// Global state to store PEMS data
let pemsState = {
    nodes: [],
    predictions: { solar: [], consumption: [], battery: [] },
    systemStatus: 'offline',
    performance: { processingTime: 0, speedup: 0, efficiency: 0 },
    logs: [],
    isRunning: false,
    mpiProcess: null
};

// Broadcast function for WebSocket
function broadcast(data) {
    wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify(data));
        }
    });
}

// REST API Endpoints

// Get current system status
app.get('/api/status', (req, res) => {
    res.json({
        status: pemsState.systemStatus,
        isRunning: pemsState.isRunning,
        nodeCount: pemsState.nodes.length,
        timestamp: new Date().toISOString()
    });
});

// Get all node data
app.get('/api/nodes', (req, res) => {
    res.json({
        nodes: pemsState.nodes,
        timestamp: new Date().toISOString()
    });
});

// Get specific node data
app.get('/api/nodes/:id', (req, res) => {
    const nodeId = parseInt(req.params.id);
    const node = pemsState.nodes.find(n => n.id === nodeId);
    
    if (node) {
        res.json(node);
    } else {
        res.status(404).json({ error: 'Node not found' });
    }
});

// Get predictions
app.get('/api/predictions', (req, res) => {
    res.json({
        predictions: pemsState.predictions,
        timestamp: new Date().toISOString()
    });
});

// Get performance metrics
app.get('/api/performance', (req, res) => {
    res.json({
        performance: pemsState.performance,
        timestamp: new Date().toISOString()
    });
});

// Get system logs
app.get('/api/logs', (req, res) => {
    res.json({
        logs: pemsState.logs.slice(-50), // Last 50 logs
        timestamp: new Date().toISOString()
    });
});

// Start PEMS system
app.post('/api/start', (req, res) => {
    if (pemsState.isRunning) {
        return res.status(400).json({ error: 'PEMS is already running' });
    }
    
    const nodeCount = req.body.nodeCount || 4;
    startPEMSProcess(nodeCount);
    
    res.json({ 
        message: 'PEMS system starting',
        nodeCount: nodeCount,
        timestamp: new Date().toISOString()
    });
});

// Stop PEMS system
app.post('/api/stop', (req, res) => {
    if (!pemsState.isRunning) {
        return res.status(400).json({ error: 'PEMS is not running' });
    }
    
    stopPEMSProcess();
    
    res.json({ 
        message: 'PEMS system stopped',
        timestamp: new Date().toISOString()
    });
});

// Run prediction
app.post('/api/predict', (req, res) => {
    if (!pemsState.isRunning) {
        return res.status(400).json({ error: 'PEMS is not running' });
    }
    
    // Send command to MPI process to run prediction
    sendCommandToMPI('RUN_PREDICTION');
    
    res.json({ 
        message: 'Prediction started',
        timestamp: new Date().toISOString()
    });
});

// Optimise system
app.post('/api/optimise', (req, res) => {
    if (!pemsState.isRunning) {
        return res.status(400).json({ error: 'PEMS is not running' });
    }
    
    sendCommandToMPI('OPTIMISE_SYSTEM');
    
    res.json({ 
        message: 'System optimisation started',
        timestamp: new Date().toISOString()
    });
});

// Simulate weather event
app.post('/api/weather-event', (req, res) => {
    const eventType = req.body.eventType || 'storm';
    const intensity = req.body.intensity || 'moderate';
    
    sendCommandToMPI(`WEATHER_EVENT:${eventType}:${intensity}`);
    
    res.json({ 
        message: `Weather event simulated: ${eventType}`,
        intensity: intensity,
        timestamp: new Date().toISOString()
    });
});

// Emergency mode
app.post('/api/emergency', (req, res) => {
    sendCommandToMPI('EMERGENCY_MODE');
    
    res.json({ 
        message: 'Emergency mode activated',
        timestamp: new Date().toISOString()
    });
});

// Function to start MPI process
function startPEMSProcess(nodeCount = 4) {
    if (pemsState.mpiProcess) {
        pemsState.mpiProcess.kill();
    }
    
    // Compile and run the MPI program
    exec('mpic++ -o pems pems.cpp -std=c++11 -I/opt/homebrew/Cellar/jsoncpp/1.9.6/include -L/opt/homebrew/Cellar/jsoncpp/1.9.6/lib -ljsoncpp', (compileError) => {
        if (compileError) {
            console.error('Compilation error:', compileError);
            addLog('ERROR', 'Failed to compile PEMS');
            return;
        }
        
        // Start the MPI process
        pemsState.mpiProcess = spawn('mpirun', ['-np', nodeCount.toString(), './pems']);
        
        pemsState.mpiProcess.stdout.on('data', (data) => {
            const output = data.toString();
            parseJSONOutput(output);
        });
        
        pemsState.mpiProcess.stderr.on('data', (data) => {
            console.error('PEMS stderr:', data.toString());
            addLog('ERROR', data.toString());
        });
        
        pemsState.mpiProcess.on('close', (code) => {
            console.log(`PEMS process exited with code ${code}`);
            pemsState.isRunning = false;
            pemsState.systemStatus = 'offline';
            addLog('INFO', 'PEMS process stopped');
            broadcast({ type: 'status', data: pemsState });
        });
        
        pemsState.isRunning = true;
        pemsState.systemStatus = 'online';
        addLog('INFO', `PEMS started with ${nodeCount} nodes`);
    });
}

// Function to stop MPI process
function stopPEMSProcess() {
    if (pemsState.mpiProcess) {
        pemsState.mpiProcess.kill('SIGTERM');
        pemsState.mpiProcess = null;
    }
    pemsState.isRunning = false;
    pemsState.systemStatus = 'offline';
    addLog('INFO', 'PEMS process stopped');
}

// Function to send commands to MPI process
function sendCommandToMPI(command) {
    if (pemsState.mpiProcess && pemsState.isRunning) {
        pemsState.mpiProcess.stdin.write(command + '\\n');
        addLog('INFO', `Command sent: ${command}`);
    }
}

// Parse JSON output from C++ MPI process
function parseJSONOutput(output) {
    const lines = output.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('JSON_OUTPUT:')) {
            try {
                const jsonStr = line.substring('JSON_OUTPUT:'.length);
                const data = JSON.parse(jsonStr);
                
                switch (data.type) {
                    case 'node_data':
                        pemsState.nodes = data.nodes;
                        addLog('INFO', 'Node data updated');
                        broadcast({ type: 'nodes', data: data.nodes });
                        break;
                        
                    case 'predictions':
                        pemsState.predictions = {
                            solar: data.solar,
                            consumption: data.consumption,
                            battery: data.battery
                        };
                        addLog('INFO', `Predictions updated, deficit risk: ${(data.deficit_risk * 100).toFixed(1)}%`);
                        broadcast({ type: 'predictions', data: pemsState.predictions });
                        break;
                        
                    case 'action_plans':
                        addLog('INFO', `Action plans distributed to ${data.actions.length} nodes`);
                        broadcast({ type: 'actions', data: data.actions });
                        break;
                        
                    case 'performance':
                        pemsState.performance = {
                            processingTime: data.processing_time,
                            speedup: data.speedup,
                            efficiency: data.efficiency
                        };
                        addLog('INFO', `Performance: ${data.processing_time}ms, ${data.speedup}x speedup`);
                        broadcast({ type: 'performance', data: pemsState.performance });
                        break;
                }
            } catch (error) {
                console.error('Error parsing JSON output:', error);
            }
        } else {
            // Regular log output
            if (line.trim()) {
                addLog('INFO', line.trim());
            }
        }
    }
}

// Add log entry
function addLog(level, message) {
    const logEntry = {
        timestamp: new Date().toISOString(),
        level: level,
        message: message
    };
    
    pemsState.logs.push(logEntry);
    
    // Keep only last 100 logs
    if (pemsState.logs.length > 100) {
        pemsState.logs = pemsState.logs.slice(-100);
    }
    
    // Broadcast to WebSocket clients
    broadcast({ type: 'log', data: logEntry });
}

// WebSocket connection handler
wss.on('connection', (ws) => {
    console.log('WebSocket client connected');
    
    // Send current state to new client
    ws.send(JSON.stringify({
        type: 'initial_state',
        data: pemsState
    }));
    
    ws.on('close', () => {
        console.log('WebSocket client disconnected');
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`PEMS REST API server running on port ${PORT}`);
    console.log(`WebSocket server running on port 3001`);
    console.log(`Dashboard available at http://localhost:${PORT}`);
    
    addLog('INFO', 'REST API server started');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('Shutting down PEMS API server...');
    stopPEMSProcess();
    process.exit(0);
});

// Export for testing
module.exports = { app, pemsState };