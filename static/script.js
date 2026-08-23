/* ==================================================
   SIGN-AI TWO-WAY COMMUNICATION SYSTEM - CLIENT JAVASCRIPT
   ================================================== */

// Configuration Parameters
const CONFIDENCE_THRESHOLD = 50.0;          // 50% minimum confidence threshold
const STABLE_PREDICTIONS_REQUIRED = 3;     // 3 matching consecutive predictions required
const SPEECH_COOLDOWN_MS = 3000;           // 3 second speech cooldown
const MEDIAPIPE_VERSION = "0.4.1675469240"; // Pinned MediaPipe Hands CDN version

// Singleton MediaPipe & Camera Flags
let hands = null;
let handsInitialized = false;
let handsInitializing = false;
let mediaPipeInitPromise = null;

let videoStream = null;
let cameraRunning = false;
let isProcessingFrame = false;
let animationFrameId = null;

// Telemetry & Lifecycle Logging Flags
let hasLoggedHandDetected = false;
let hasLoggedSequenceSent = false;

// Speech Recognition Variables (Section 1)
let speechRecognition = null;
let isListening = false;

// Sliding Landmark & Prediction Buffers
let posLandmarkBuffer = []; // Max 30 frames of 126D landmarks
let recentPredictionsHistory = [];
let lastSpokenSentence = "";
let lastSpokenTimestamp = 0;
let isDebugMode = false;

// DOM Elements
const startMicBtn = document.getElementById('startMicBtn');
const stopMicBtn = document.getElementById('stopMicBtn');
const micStatusPill = document.getElementById('micStatusPill');
const micStatusText = document.getElementById('micStatusText');
const teacherSpeechDisplay = document.getElementById('teacherSpeechDisplay');

const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const cameraOverlayBadge = document.getElementById('cameraOverlayBadge');

const startCamBtn = document.getElementById('startCamBtn');
const stopCamBtn = document.getElementById('stopCamBtn');

const detectedSentenceEl = document.getElementById('detectedSentence');
const confidenceBadgeEl = document.getElementById('confidenceBadge');
const confidenceTextEl = document.getElementById('confidenceText');
const predictionsListEl = document.getElementById('predictionsList');

// Debug Panel Elements
const debugCard = document.getElementById('debugCard');
const debugMpInit = document.getElementById('debugMpInit');
const debugCamRunning = document.getElementById('debugCamRunning');
const debugVidDims = document.getElementById('debugVidDims');
const debugHandDetected = document.getElementById('debugHandDetected');
const debugFramesCollected = document.getElementById('debugFramesCollected');
const debugFeatureDim = document.getElementById('debugFeatureDim');
const debugFeatureShape = document.getElementById('debugFeatureShape');
const debugBackendStatus = document.getElementById('debugBackendStatus');
const debugRawPred = document.getElementById('debugRawPred');
const debugConfidence = document.getElementById('debugConfidence');
const debugStableCount = document.getElementById('debugStableCount');
const debugInferenceTime = document.getElementById('debugInferenceTime');

// Initialize on Page Load
document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
    initializeMediaPipe().catch(err => {
        console.warn("[DOMContentLoaded] Initial MediaPipe load deferred:", err);
    });
});

/* ==================================================
   SECTION 1 — TEACHER VOICE (WEB SPEECH API)
   ================================================== */
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        micStatusText.innerText = "Speech API Not Supported";
        micStatusPill.querySelector('.dot-indicator').className = "dot-indicator red";
        startMicBtn.disabled = true;
        return;
    }

    speechRecognition = new SpeechRecognition();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;
    speechRecognition.lang = 'en-US';

    speechRecognition.onstart = () => {
        isListening = true;
        startMicBtn.disabled = true;
        stopMicBtn.disabled = false;
        micStatusText.innerText = "Listening...";
        micStatusPill.querySelector('.dot-indicator').className = "dot-indicator red";
    };

    speechRecognition.onend = () => {
        isListening = false;
        startMicBtn.disabled = false;
        stopMicBtn.disabled = true;
        micStatusText.innerText = "Ready";
        micStatusPill.querySelector('.dot-indicator').className = "dot-indicator green";
    };

    speechRecognition.onerror = (event) => {
        console.warn("[SpeechRecognition Error]", event.error);
        if (event.error !== 'no-speech') {
            micStatusText.innerText = "Mic Error: " + event.error;
        }
    };

    speechRecognition.onresult = (event) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            finalTranscript += event.results[i][0].transcript;
        }
        if (finalTranscript.trim() !== "") {
            teacherSpeechDisplay.innerHTML = `"${finalTranscript.trim()}"`;
        }
    };
}

function startSpeechRecognition() {
    if (speechRecognition && !isListening) {
        try {
            speechRecognition.start();
        } catch (e) {
            console.error("Failed to start speech recognition", e);
        }
    }
}

function stopSpeechRecognition() {
    if (speechRecognition && isListening) {
        speechRecognition.stop();
    }
}

function clearTeacherText() {
    teacherSpeechDisplay.innerHTML = '"Waiting for teacher to speak..."';
}

/* ==================================================
   SECTION 2 — MEDIAPIPE INITIALIZATION & CAMERA
   ================================================== */
function updateDetectionStatus(colorClass, text) {
    const statusDot = document.getElementById('detectionDot');
    const statusText = document.getElementById('detectionStatusText');
    if (statusDot) statusDot.className = `dot-indicator ${colorClass}`;
    if (statusText) statusText.innerText = text;
}

async function initializeMediaPipe() {
    if (handsInitialized && hands) {
        return hands;
    }
    if (handsInitializing && mediaPipeInitPromise) {
        return mediaPipeInitPromise;
    }

    handsInitializing = true;
    console.log("[INIT] MediaPipe starting");
    updateDetectionStatus("yellow", "Initializing MediaPipe...");
    if (debugMpInit) debugMpInit.innerText = "LOADING...";

    mediaPipeInitPromise = new Promise((resolve, reject) => {
        try {
            if (typeof Hands === 'undefined') {
                throw new Error("MediaPipe Hands library (hands.js) script tag not loaded.");
            }

            hands = new Hands({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${MEDIAPIPE_VERSION}/${file}`
            });

            hands.setOptions({
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.3,
                minTrackingConfidence: 0.3
            });

            hands.onResults(onMediaPipeResults);

            handsInitialized = true;
            handsInitializing = false;
            console.log("[INIT] MediaPipe ready");
            updateDetectionStatus("green", "MediaPipe Ready");
            if (debugMpInit) debugMpInit.innerText = "YES";
            resolve(hands);
        } catch (err) {
            console.error("[INIT ERROR] MediaPipe initialization failed:", err);
            handsInitialized = false;
            handsInitializing = false;
            updateDetectionStatus("red", "MediaPipe Init Failed");
            if (debugMpInit) debugMpInit.innerText = "ERROR";
            reject(err);
        }
    });

    return mediaPipeInitPromise;
}

async function startCamera() {
    if (cameraRunning) {
        console.log("[CAMERA] Start Camera ignored: already running.");
        return;
    }

    try {
        await initializeMediaPipe();

        updateDetectionStatus("yellow", "Starting Camera...");
        const constraints = {
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };

        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamElement.srcObject = videoStream;
        await webcamElement.play();

        cameraRunning = true;
        hasLoggedHandDetected = false;
        hasLoggedSequenceSent = false;

        console.log("[CAMERA] Stream started");
        console.log("[CAMERA] Processing loop started");

        startCamBtn.disabled = true;
        stopCamBtn.disabled = false;
        cameraPlaceholder.style.display = 'none';
        cameraOverlayBadge.style.display = 'flex';

        canvasElement.width = webcamElement.videoWidth || 640;
        canvasElement.height = webcamElement.videoHeight || 480;

        if (debugCamRunning) debugCamRunning.innerText = "YES";
        if (debugVidDims) debugVidDims.innerText = `${canvasElement.width} × ${canvasElement.height}`;

        updateDetectionStatus("green", "Camera Active");
        processVideoFrameLoop();
    } catch (err) {
        console.error("[CAMERA ERROR] Failed to start camera:", err);
        cameraRunning = false;
        alert(`Unable to access camera: ${err.message || 'Permission denied'}`);
        updateDetectionStatus("red", "Camera Error");
        if (debugCamRunning) debugCamRunning.innerText = "NO";
    }
}

function stopCamera() {
    cameraRunning = false;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }

    webcamElement.srcObject = null;
    isProcessingFrame = false;

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    startCamBtn.disabled = false;
    stopCamBtn.disabled = true;
    cameraPlaceholder.style.display = 'flex';
    cameraOverlayBadge.style.display = 'none';

    posLandmarkBuffer = [];
    recentPredictionsHistory = [];
    detectedSentenceEl.innerText = '"Waiting for sign..."';
    confidenceTextEl.innerText = '-- %';

    if (debugCamRunning) debugCamRunning.innerText = "NO";
    if (debugHandDetected) debugHandDetected.innerText = "NO";
    if (debugFramesCollected) debugFramesCollected.innerText = "0 / 30";

    console.log("[CAMERA] Processing loop stopped");
    updateDetectionStatus("yellow", "Camera Stopped");
}

async function processVideoFrameLoop() {
    if (!cameraRunning) return;

    if (!isProcessingFrame && webcamElement.readyState >= 2 && hands) {
        isProcessingFrame = true;
        try {
            await hands.send({ image: webcamElement });
        } catch (err) {
            console.error("[MediaPipe Send Error]", err);
        } finally {
            isProcessingFrame = false;
        }
    }

    animationFrameId = requestAnimationFrame(processVideoFrameLoop);
}

/* ==================================================
   MEDIAPIPE LANDMARK PROCESSING & 252D INFERENCE
   ================================================== */
function onMediaPipeResults(results) {
    if (!cameraRunning) return;

    // Sync canvas resolution with webcam video width/height if needed
    if (webcamElement.videoWidth && canvasElement.width !== webcamElement.videoWidth) {
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
    }

    // Clear canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Multi-Person Active Signer Tracking State
    let leftHand = new Array(63).fill(0.0);
    let rightHand = new Array(63).fill(0.0);
    let handsDetected = 0;
    let peopleCount = 0;
    let selectedSignerCenter = null;

    if (results.multiHandLandmarks && results.multiHandedness) {
        handsDetected = results.multiHandLandmarks.length;

        // Resolve connections safely from global scope
        const connections = (typeof HAND_CONNECTIONS !== 'undefined') ? HAND_CONNECTIONS : (window.HAND_CONNECTIONS || []);

        if (handsDetected === 0) {
            peopleCount = 0;
        } else if (handsDetected === 1) {
            peopleCount = 1;
            const lm = results.multiHandLandmarks[0];
            selectedSignerCenter = { x: lm[9].x, y: lm[9].y };
        } else if (handsDetected >= 2) {
            // Check inter-wrist distance to cluster hands into people
            const lm1 = results.multiHandLandmarks[0];
            const lm2 = results.multiHandLandmarks[1];
            const dx = lm1[0].x - lm2[0].x;
            const dy = lm1[0].y - lm2[0].y;
            const wristDist = Math.sqrt(dx * dx + dy * dy);

            if (wristDist <= 0.45) {
                // Both hands belong to 1 single person
                peopleCount = 1;
                selectedSignerCenter = { x: (lm1[9].x + lm2[9].x) / 2.0, y: (lm1[9].y + lm2[9].y) / 2.0 };
            } else {
                // Hands belong to 2 distinct people -> Select Primary Signer based on tracking & center proximity
                peopleCount = 2;
                
                // Score Candidate 1 vs Candidate 2
                const c1 = { x: lm1[9].x, y: lm1[9].y };
                const c2 = { x: lm2[9].x, y: lm2[9].y };

                let score1 = 1.0 - (Math.abs(c1.x - 0.5) + Math.abs(c1.y - 0.5));
                let score2 = 1.0 - (Math.abs(c2.x - 0.5) + Math.abs(c2.y - 0.5));

                if (window.lastTrackedSignerCenter) {
                    const d1 = Math.sqrt(Math.pow(c1.x - window.lastTrackedSignerCenter.x, 2) + Math.pow(c1.y - window.lastTrackedSignerCenter.y, 2));
                    const d2 = Math.sqrt(Math.pow(c2.x - window.lastTrackedSignerCenter.x, 2) + Math.pow(c2.y - window.lastTrackedSignerCenter.y, 2));
                    score1 += (1.0 - d1) * 2.0; // Priority: Continuity
                    score2 += (1.0 - d2) * 2.0;
                }

                selectedSignerCenter = (score1 >= score2) ? c1 : c2;
            }
        }

        // Check if active signer switched abruptly (> 0.35 screen distance)
        if (selectedSignerCenter && window.lastTrackedSignerCenter) {
            const shiftDist = Math.sqrt(
                Math.pow(selectedSignerCenter.x - window.lastTrackedSignerCenter.x, 2) +
                Math.pow(selectedSignerCenter.y - window.lastTrackedSignerCenter.y, 2)
            );
            if (shiftDist > 0.35) {
                console.log("[TRACKING] Active signer changed — Resetting 30-frame sequence buffer");
                posLandmarkBuffer = [];
                recentPredictionsHistory = [];
            }
        }
        if (selectedSignerCenter) {
            window.lastTrackedSignerCenter = selectedSignerCenter;
        }

        // Filter and process landmarks for selected primary signer
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];
            const classification = results.multiHandedness[i].label; // 'Left' or 'Right'
            const handCenter = { x: landmarks[9].x, y: landmarks[9].y };

            // Ignore hands from secondary people when multiple people are detected
            if (peopleCount >= 2 && selectedSignerCenter) {
                const distToSelected = Math.sqrt(
                    Math.pow(handCenter.x - selectedSignerCenter.x, 2) +
                    Math.pow(handCenter.y - selectedSignerCenter.y, 2)
                );
                if (distToSelected > 0.35) {
                    continue; // Skip secondary person hand
                }
            }

            // Draw MediaPipe landmarks safely
            try {
                if (typeof drawConnectors === 'function' && connections.length > 0) {
                    drawConnectors(canvasCtx, landmarks, connections, { color: '#6366f1', lineWidth: 3 });
                }
                if (typeof drawLandmarks === 'function') {
                    drawLandmarks(canvasCtx, landmarks, { color: '#10b981', lineWidth: 2, radius: 4 });
                }
            } catch (drawErr) {
                console.warn("[MediaPipe Draw Warning]", drawErr);
            }

            // Extract 63 (x,y,z) coordinates per hand
            const coords = [];
            landmarks.forEach(lm => {
                coords.push(lm.x, lm.y, lm.z);
            });

            if (classification === 'Left') {
                leftHand = coords;
            } else {
                rightHand = coords;
            }
        }
    }
    canvasCtx.restore();

    if (debugHandDetected) {
        debugHandDetected.innerText = handsDetected > 0 ? `YES (${peopleCount} signer)` : "NO";
    }

    // Buffer 126D landmark vector for primary active signer
    if (handsDetected > 0) {
        const frame126 = leftHand.concat(rightHand);
        posLandmarkBuffer.push(frame126);

        if (posLandmarkBuffer.length > 30) {
            posLandmarkBuffer.shift();
        }

        if (peopleCount >= 2) {
            updateDetectionStatus("yellow", "Multiple people detected — tracking selected signer");
        } else {
            updateDetectionStatus("green", "Single signer detected");
        }
    } else {
        if (posLandmarkBuffer.length > 0) {
            posLandmarkBuffer.shift();
        }
        if (posLandmarkBuffer.length === 0) {
            updateDetectionStatus("yellow", "No signer detected");
        } else {
            updateDetectionStatus("yellow", `Collecting frames: ${posLandmarkBuffer.length} / 30`);
        }
    }

    if (debugFramesCollected) {
        debugFramesCollected.innerText = `${posLandmarkBuffer.length} / 30`;
    }

    // Trigger 252D PyTorch Model Inference when buffer reaches 30 frames
    if (posLandmarkBuffer.length === 30) {
        if (!hasLoggedSequenceSent) {
            console.log("[MODEL] Sending 30-frame sequence");
            hasLoggedSequenceSent = true;
        }
        sendInferenceRequest([...posLandmarkBuffer]);
    } else {
        hasLoggedSequenceSent = false;
    }
}

async function sendInferenceRequest(sequence30x126) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: sequence30x126 })
        });

        if (!response.ok) {
            if (debugBackendStatus) debugBackendStatus.innerText = "ERROR " + response.status;
            return;
        }

        const data = await response.json();
        if (debugBackendStatus) debugBackendStatus.innerText = "CONNECTED";
        updateUIWithPrediction(data);
    } catch (err) {
        console.warn("[Inference API Error]", err);
        if (debugBackendStatus) debugBackendStatus.innerText = "ERROR";
    }
}

function updateUIWithPrediction(data) {
    const predictedText = data.predicted_text || data.sentence || "Not confident. Please repeat the sign.";
    const confidence = data.confidence || 0.0;
    const topPreds = data.top_predictions || [];
    const isStable = (data.stable === true);
    const isUncertain = (data.uncertain === true);
    const probMargin = data.prob_margin || 0.0;
    const voteCount = data.vote_count || 0;
    const infTimeMs = data.inference_time_ms || 0;

    console.log("[MODEL] Prediction:", predictedText, `(Conf: ${confidence}%, Margin: ${probMargin}%, Stable: ${isStable}, Uncertain: ${isUncertain})`);

    // 1. Update Top Predictions List
    updatePredictionsListUI(topPreds);

    // 2. Update Debug Telemetry
    if (debugCard) {
        if (debugRawPred) debugRawPred.innerText = predictedText;
        if (debugConfidence) debugConfidence.innerText = `${confidence}% (Margin: ${probMargin}%)`;
        if (debugStableCount) debugStableCount.innerText = `${voteCount} votes | Stable: ${isStable}`;
        if (debugInferenceTime) debugInferenceTime.innerText = `${infTimeMs} ms`;
        if (debugFeatureDim) debugFeatureDim.innerText = "252";
        if (debugFeatureShape) debugFeatureShape.innerText = "(30, 252)";
    }

    // 3. Update Main UI Display
    if (isStable && !isUncertain && predictedText !== "Not confident. Please repeat the sign.") {
        const uppercaseSentence = predictedText.toUpperCase();
        detectedSentenceEl.innerText = `"${uppercaseSentence}"`;
        confidenceTextEl.innerText = `${confidence.toFixed(1)}%`;
        updateDetectionStatus("green", "Prediction Ready");

        // Speech Cooldown Trigger (3.0s)
        const now = Date.now();
        if (now - lastSpokenTimestamp >= SPEECH_COOLDOWN_MS && predictedText !== lastSpokenSentence) {
            speakSentenceTTS(predictedText);
            lastSpokenSentence = predictedText;
            lastSpokenTimestamp = now;
        }
    } else {
        detectedSentenceEl.innerText = '"Not confident. Please repeat the sign."';
        confidenceTextEl.innerText = `${confidence.toFixed(1)}%`;
        updateDetectionStatus("yellow", `Not confident (${confidence.toFixed(0)}%)`);
    }
}

function updatePredictionsListUI(topPreds) {
    if (!predictionsListEl) return;
    predictionsListEl.innerHTML = '';

    if (topPreds.length === 0) {
        predictionsListEl.innerHTML = '<li class="prediction-item"><span>1. Waiting...</span><span class="prob-val">--%</span></li>';
        return;
    }

    topPreds.slice(0, 3).forEach((pred, idx) => {
        const li = document.createElement('li');
        li.className = 'prediction-item';
        li.innerHTML = `
            <span>${idx + 1}. ${pred.sentence.toUpperCase()}</span>
            <span class="prob-val">${pred.confidence.toFixed(1)}%</span>
        `;
        predictionsListEl.appendChild(li);
    });
}

function speakSentenceTTS(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();

    const cleanText = text.replace(/[^a-zA-Z0-9\s]/g, '').trim();
    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
}

/* ==================================================
   DEBUG MODE TOGGLE
   ================================================== */
function toggleDebugMode() {
    isDebugMode = !isDebugMode;
    if (debugCard) {
        debugCard.style.display = isDebugMode ? 'block' : 'none';
    }
}
