/**
 * TRUTHSCAN AI - Master Client Application Logic
 * Single-Page Web Application Controller for Localhost & Vercel
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSystemStatus();
    initDetector();
    initModelComparison();
    initTrainingSession();
    initSessions();
});

/* ==========================================================================
   1. NAVIGATION ROUTER (Zero Ghosting / Strict Active Element Rendering)
   ========================================================================== */
function initNavigation() {
    const tiles = document.querySelectorAll('.nav-tile');
    tiles.forEach(tile => {
        tile.addEventListener('click', () => {
            const targetPageId = tile.getAttribute('data-page');
            switchPage(targetPageId);
        });
    });

    const startBtn = document.getElementById('btn-start-detection');
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            switchPage('page-detector');
        });
    }
}

function switchPage(pageId) {
    // 1. Remove active state from all pages and nav tiles
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.querySelectorAll('.nav-tile').forEach(tile => {
        tile.classList.remove('active');
        tile.setAttribute('aria-selected', 'false');
    });

    // 2. Activate selected page and tile
    const activePage = document.getElementById(pageId);
    if (activePage) {
        activePage.classList.add('active');
    }
    const activeTile = document.querySelector(`.nav-tile[data-page="${pageId}"]`);
    if (activeTile) {
        activeTile.classList.add('active');
        activeTile.setAttribute('aria-selected', 'true');
    }

    // 3. Trigger page-specific data refreshes
    if (pageId === 'page-comparison') {
        loadModelComparison();
    } else if (pageId === 'page-training') {
        loadDatasetStats();
    } else if (pageId === 'page-sessions') {
        renderPredictionHistory();
        loadSessionsData();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ==========================================================================
   2. SYSTEM STATUS STRIP
   ========================================================================== */
async function initSystemStatus() {
    try {
        const resp = await fetch('/api/status');
        if (!resp.ok) throw new Error('Status endpoint unavailable');
        const data = await resp.json();

        const datasetElem = document.getElementById('status-dataset');
        if (datasetElem) {
            datasetElem.textContent = data.dataset_available ? 'Available (ISOT)' : 'Not Found';
            datasetElem.className = `status-pill ${data.dataset_available ? 'status-success' : 'status-danger'}`;
        }

        const computeElem = document.getElementById('status-compute');
        if (computeElem) computeElem.textContent = data.compute_device || 'CPU';

        const cnnElem = document.getElementById('status-cnn');
        if (cnnElem) {
            cnnElem.textContent = data.cnn_status;
            cnnElem.className = `status-pill ${data.cnn_status === 'Trained' ? 'status-success' : ''}`;
        }

        const lstmElem = document.getElementById('status-lstm');
        if (lstmElem) {
            lstmElem.textContent = data.lstm_status;
            lstmElem.className = `status-pill ${data.lstm_status === 'Trained' ? 'status-success' : ''}`;
        }

        const specCnn = document.getElementById('specs-cnn-status');
        if (specCnn) {
            specCnn.textContent = data.cnn_status;
            specCnn.className = `status-pill ${data.cnn_status === 'Trained' ? 'status-success' : 'status-danger'}`;
        }

        const specLstm = document.getElementById('specs-lstm-status');
        if (specLstm) {
            specLstm.textContent = data.lstm_status;
            specLstm.className = `status-pill ${data.lstm_status === 'Trained' ? 'status-success' : 'status-danger'}`;
        }
    } catch (e) {
        console.warn('Could not fetch status:', e);
    }
}

/* ==========================================================================
   3. FAKE NEWS DETECTOR
   ========================================================================== */
function initDetector() {
    const analyzeBtn = document.getElementById('btn-analyze');
    const headlineInput = document.getElementById('input-headline');
    const articleInput = document.getElementById('input-article');
    const modelSelect = document.getElementById('select-model');
    const resultsContainer = document.getElementById('detector-results');

    // Sample Fill Helpers
    const sampleRealBtn = document.getElementById('btn-sample-real');
    if (sampleRealBtn) {
        sampleRealBtn.addEventListener('click', () => {
            headlineInput.value = "Senate leaders reach bipartisan agreement on federal infrastructure spending";
            articleInput.value = "Congressional committee leaders announced a compromise package authorizing funding for highway repairs, municipal bridges, and public transit modernization across several states. Lawmakers praised the negotiations as an essential step toward economic resilience.";
        });
    }

    const sampleFakeBtn = document.getElementById('btn-sample-fake');
    if (sampleFakeBtn) {
        sampleFakeBtn.addEventListener('click', () => {
            headlineInput.value = "BOMBSHELL: Leaked documents prove deep state conspiracy to confiscate personal savings!";
            articleInput.value = "Shocking leaked communications from globalist elites reveal an imminent emergency order to freeze all citizen accounts. Mainstream media has maintained total blackout on this explosive revelation that changes everything!";
        });
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            const title = headlineInput.value.trim();
            const text = articleInput.value.trim();
            const model = modelSelect.value;

            if (!title && !text) {
                alert('Please enter a headline or article text to analyze.');
                return;
            }

            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'ANALYZING...';
            resultsContainer.style.display = 'block';
            resultsContainer.innerHTML = '<div class="loading-spinner">Executing neural preprocessing and deep learning inference...</div>';

            try {
                const resp = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, text, model })
                });

                const data = await resp.json();
                if (!resp.ok) {
                    throw new Error(data.error || 'Prediction request failed');
                }

                renderPredictionOutput(data, title, text, model);
                recordPredictionSession(data, title, text);

            } catch (err) {
                resultsContainer.innerHTML = `<div class="alert alert-warning">Inference error: ${err.message}</div>`;
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'ANALYZE NEWS';
            }
        });
    }
}

function renderPredictionOutput(data, title, text, modelChoice) {
    const container = document.getElementById('detector-results');
    const isBoth = (modelChoice === 'both');

    if (isBoth) {
        const cnn = data.cnn;
        const lstm = data.lstm;

        const cnnIsFake = (cnn.prediction === 'FAKE NEWS');
        const lstmIsFake = (lstm.prediction === 'FAKE NEWS');

        container.innerHTML = `
            <hr class="divider">
            <h3 class="page-title" style="font-size: 1.25rem;">Side-by-Side Model Prediction</h3>
            <div class="grid-2col" style="margin-top: 1rem;">
                <div class="compare-card">
                    <div class="compare-model-name">Convolutional Neural Network (CNN)</div>
                    <div class="pred-header-kicker">Model Prediction</div>
                    <div class="${cnnIsFake ? 'pred-main-fake' : 'pred-main-real'}">${cnnIsFake ? 'FAKE' : 'REAL'}</div>
                    <div class="pred-conf-block">
                        <span class="pred-conf-caption">Confidence</span>
                        <span class="pred-conf-number">${cnn.confidence.toFixed(1)}%</span>
                    </div>
                </div>
                <div class="compare-card">
                    <div class="compare-model-name">Long Short-Term Memory (LSTM)</div>
                    <div class="pred-header-kicker">Model Prediction</div>
                    <div class="${lstmIsFake ? 'pred-main-fake' : 'pred-main-real'}">${lstmIsFake ? 'FAKE' : 'REAL'}</div>
                    <div class="pred-conf-block">
                        <span class="pred-conf-caption">Confidence</span>
                        <span class="pred-conf-number">${lstm.confidence.toFixed(1)}%</span>
                    </div>
                </div>
            </div>

            <div class="section-title" style="margin-top: 1.8rem;">Comparison Summary</div>
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                            <th>Raw Fake Probability</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.comparison.map(row => `
                            <tr>
                                <td><strong>${row.Model}</strong></td>
                                <td><span style="color: ${row.Prediction.includes('FAKE') ? 'var(--danger-red)' : 'var(--accent-green)'}; font-weight: 700;">${row.Prediction}</span></td>
                                <td>${row.Confidence}</td>
                                <td>${row['Raw Fake Probability']}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            <div class="page-caption" style="margin-top: 0.5rem; font-size: 0.8rem;">
                Note: Predictions reflect statistical and lexical patterns learned from the training corpus. The system does not query live web sources.
            </div>
        `;
    } else {
        const isFake = (data.prediction === 'FAKE NEWS');
        container.innerHTML = `
            <hr class="divider">
            <div class="prediction-container">
                <div class="pred-header-kicker">Model Prediction (${data.model})</div>
                <div class="${isFake ? 'pred-main-fake' : 'pred-main-real'}">${isFake ? 'FAKE' : 'REAL'}</div>
                <div class="pred-conf-block">
                    <span class="pred-conf-caption">Confidence</span>
                    <span class="pred-conf-number">${data.confidence.toFixed(1)}%</span>
                </div>
            </div>
            <div class="page-caption" style="text-align: center; margin-top: 0.5rem;">
                Architecture: ${data.model} | Evaluated Sequence Word Count: ${data.word_count} words
            </div>
        `;
    }
}

/* ==========================================================================
   4. MODEL COMPARISON
   ========================================================================== */
async function loadModelComparison() {
    const loadingElem = document.getElementById('comparison-loading');
    const contentElem = document.getElementById('comparison-content');
    const bestBanner = document.getElementById('best-model-banner');
    const tbody = document.getElementById('tbody-comparison');

    try {
        const resp = await fetch('/api/comparison');
        const data = await resp.json();

        if (!data.available || !data.records || data.records.length === 0) {
            loadingElem.innerHTML = `
                <div class="alert alert-warning">
                    Models not evaluated yet. Please navigate to the <strong>Training Session</strong> tab to train models.
                </div>
            `;
            contentElem.style.display = 'none';
            return;
        }

        loadingElem.style.display = 'none';
        contentElem.style.display = 'block';

        // Best model banner
        if (data.best_model) {
            bestBanner.innerHTML = `★ <strong>Highest Performing Architecture:</strong> ${data.best_model} (F1 Score: ${(data.best_f1 * 100).toFixed(2)}%)`;
            bestBanner.style.display = 'block';
        } else {
            bestBanner.style.display = 'none';
        }

        // Summary table rows
        tbody.innerHTML = data.records.map(r => `
            <tr>
                <td><strong>${r.Model}</strong></td>
                <td>${(r.Accuracy * 100).toFixed(2)}%</td>
                <td>${(r.Precision * 100).toFixed(2)}%</td>
                <td>${(r.Recall * 100).toFixed(2)}%</td>
                <td><strong>${(r['F1 Score'] * 100).toFixed(2)}%</strong></td>
                <td>${r['Training Time'] || 'N/A'}</td>
            </tr>
        `).join('');

        // Refresh plot images with cache-busting timestamp
        const t = Date.now();
        const imgComp = document.getElementById('img-plot-comparison');
        const imgCurves = document.getElementById('img-plot-curves');
        const imgCm = document.getElementById('img-plot-cm');

        if (imgComp) imgComp.src = `${data.plots.comparison}?t=${t}`;
        if (imgCurves) imgCurves.src = `${data.plots.learning_curves}?t=${t}`;
        if (imgCm) imgCm.src = `${data.plots.confusion_matrices}?t=${t}`;

    } catch (err) {
        loadingElem.innerHTML = `<div class="alert alert-warning">Failed to load comparison data: ${err.message}</div>`;
    }
}

function initModelComparison() {
    // Initial call will occur when tab is clicked
}

/* ==========================================================================
   5. TRAINING SESSION
   ========================================================================== */
async function loadDatasetStats() {
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();

        if (data.available) {
            const elTotal = document.getElementById('stat-total');
            const elFake = document.getElementById('stat-fake');
            const elReal = document.getElementById('stat-real');

            if (elTotal) elTotal.textContent = Number(data.total_articles).toLocaleString();
            if (elFake) elFake.textContent = Number(data.fake_articles).toLocaleString();
            if (elReal) elReal.textContent = Number(data.real_articles).toLocaleString();
        }
    } catch (e) {
        console.warn('Could not load dataset stats:', e);
    }
}

function initTrainingSession() {
    const radioModes = document.querySelectorAll('input[name="training-mode"]');
    const paramsList = document.getElementById('params-list');
    const confirmCheckbox = document.getElementById('chk-confirm-train');

    const btnTrainCnn = document.getElementById('btn-train-cnn');
    const btnTrainLstm = document.getElementById('btn-train-lstm');
    const btnTrainBoth = document.getElementById('btn-train-both');

    const progressContainer = document.getElementById('training-progress-container');
    const progressBar = document.getElementById('training-progress-bar');
    const statusText = document.getElementById('training-status-text');
    const consoleLogs = document.getElementById('training-console-logs');
    const latestContainer = document.getElementById('latest-training-container');

    // Preset selection change
    radioModes.forEach(radio => {
        radio.addEventListener('change', () => {
            const mode = radio.value;
            if (mode === 'quick') {
                paramsList.innerHTML = `
                    <li><strong>Samples:</strong> 5,000 authentic articles</li>
                    <li><strong>Epochs:</strong> 2</li>
                    <li><strong>Batch Size:</strong> 16</li>
                    <li><strong>Learning Rate:</strong> 0.001</li>
                    <li><strong>Split:</strong> 80% Train, 10% Validation, 10% Test (Stratified)</li>
                `;
            } else {
                paramsList.innerHTML = `
                    <li><strong>Samples:</strong> 20,000 authentic articles</li>
                    <li><strong>Epochs:</strong> 5</li>
                    <li><strong>Batch Size:</strong> 32</li>
                    <li><strong>Learning Rate:</strong> 0.001</li>
                    <li><strong>Split:</strong> 80% Train, 10% Validation, 10% Test (Stratified)</li>
                `;
            }
        });
    });

    // Enable/disable buttons based on confirm checkbox
    if (confirmCheckbox) {
        confirmCheckbox.addEventListener('change', () => {
            const isConfirmed = confirmCheckbox.checked;
            btnTrainCnn.disabled = !isConfirmed;
            btnTrainLstm.disabled = !isConfirmed;
            btnTrainBoth.disabled = !isConfirmed;
        });
    }

    // Launch Training Function
    async function launchTraining(target) {
        const selectedMode = document.querySelector('input[name="training-mode"]:checked').value;

        btnTrainCnn.disabled = true;
        btnTrainLstm.disabled = true;
        btnTrainBoth.disabled = true;

        progressContainer.style.display = 'block';
        progressBar.style.width = '10%';
        statusText.textContent = `Starting training session for ${target.toUpperCase()} (${selectedMode.toUpperCase()} mode)...`;
        consoleLogs.textContent = `[TRUTHSCAN AI] Initializing training pipeline for ${target.toUpperCase()}...\n[TRUTHSCAN AI] Loading dataset and building vocabulary...`;

        // Progress simulation for responsive UX
        let currentPct = 15;
        const progressTimer = setInterval(() => {
            if (currentPct < 85) {
                currentPct += 5;
                progressBar.style.width = `${currentPct}%`;
                statusText.textContent = `Training ${target.toUpperCase()} in progress (${currentPct}%)...`;
            }
        }, 1500);

        try {
            const resp = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target, mode: selectedMode })
            });

            clearInterval(progressTimer);
            const data = await resp.json();

            if (!resp.ok) throw new Error(data.error || 'Training process failed');

            progressBar.style.width = '100%';
            statusText.textContent = `Training and evaluation complete for ${target.toUpperCase()}!`;
            
            if (data.logs && data.logs.length > 0) {
                consoleLogs.textContent = data.logs.join('\n');
                consoleLogs.scrollTop = consoleLogs.scrollHeight;
            }

            // Render persistent latest training results
            renderLatestTrainingBox(target, selectedMode, data);

            // Re-fetch system status and comparison data
            initSystemStatus();

        } catch (err) {
            clearInterval(progressTimer);
            progressBar.style.width = '100%';
            progressBar.style.background = 'var(--danger-red)';
            statusText.textContent = `Training error: ${err.message}`;
            consoleLogs.textContent += `\n[ERROR] ${err.message}`;
        } finally {
            if (confirmCheckbox.checked) {
                btnTrainCnn.disabled = false;
                btnTrainLstm.disabled = false;
                btnTrainBoth.disabled = false;
            }
        }
    }

    if (btnTrainCnn) btnTrainCnn.addEventListener('click', () => launchTraining('cnn'));
    if (btnTrainLstm) btnTrainLstm.addEventListener('click', () => launchTraining('lstm'));
    if (btnTrainBoth) btnTrainBoth.addEventListener('click', () => launchTraining('both'));
}

function renderLatestTrainingBox(target, mode, data) {
    const container = document.getElementById('latest-training-container');
    const timestamp = new Date().toLocaleTimeString();

    container.style.display = 'block';
    container.innerHTML = `
        <hr class="divider">
        <h3 class="page-title" style="font-size: 1.25rem;">Latest Training Session Output (${target.toUpperCase()})</h3>
        <div class="page-caption">Completed at ${timestamp} • Mode: ${mode.toUpperCase()}</div>

        <div class="section-title">Updated Test Evaluation Results</div>
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Training Time</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.comparison ? data.comparison.map(r => `
                        <tr>
                            <td><strong>${r.Model}</strong></td>
                            <td>${(r.Accuracy * 100).toFixed(2)}%</td>
                            <td>${(r.Precision * 100).toFixed(2)}%</td>
                            <td>${(r.Recall * 100).toFixed(2)}%</td>
                            <td><strong>${(r['F1 Score'] * 100).toFixed(2)}%</strong></td>
                            <td>${r['Training Time'] || 'N/A'}</td>
                        </tr>
                    `).join('') : '<tr><td colspan="6">Evaluation metrics updated.</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="chart-box" style="margin-top: 1.2rem;">
            <div class="chart-caption">Updated Model Comparison on Test Split</div>
            <img src="/api/plots/model_comparison.png?t=${Date.now()}" class="chart-img" alt="Updated Comparison">
        </div>
    `;
}

/* ==========================================================================
   6. SESSIONS & HISTORICAL ANALYSIS
   ========================================================================== */
function initSessions() {
    // Subtab switching
    const subtabs = document.querySelectorAll('.subtab-btn');
    subtabs.forEach(tab => {
        tab.addEventListener('click', () => {
            subtabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const targetId = tab.getAttribute('data-subtab');
            const content = document.getElementById(targetId);
            if (content) content.classList.add('active');
        });
    });

    // Clear prediction history button
    const clearBtn = document.getElementById('btn-clear-history');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (confirm('Clear all recorded prediction history for this session?')) {
                localStorage.removeItem('truthscan_history');
                renderPredictionHistory();
            }
        });
    }
}

function getStoredPredictions() {
    try {
        const raw = localStorage.getItem('truthscan_history');
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

function recordPredictionSession(predData, title, text) {
    const history = getStoredPredictions();
    const timestamp = new Date().toLocaleString();
    const snippet = title ? title : (text.slice(0, 60) + '...');

    let resultSummary = '';
    if (predData.cnn && predData.lstm) {
        resultSummary = `CNN: ${predData.cnn.prediction} (${predData.cnn.confidence.toFixed(1)}%) | LSTM: ${predData.lstm.prediction} (${predData.lstm.confidence.toFixed(1)}%)`;
    } else {
        resultSummary = `${predData.prediction} (${predData.confidence.toFixed(1)}%)`;
    }

    history.unshift({
        timestamp,
        snippet,
        title,
        text,
        model: predData.model || 'Both (CNN + LSTM)',
        result: resultSummary,
        cnn_pred: predData.cnn ? predData.cnn.prediction : (predData.model === 'CNN' ? predData.prediction : 'N/A'),
        cnn_conf: predData.cnn ? `${predData.cnn.confidence.toFixed(1)}%` : (predData.model === 'CNN' ? `${predData.confidence.toFixed(1)}%` : 'N/A'),
        lstm_pred: predData.lstm ? predData.lstm.prediction : (predData.model === 'LSTM' ? predData.prediction : 'N/A'),
        lstm_conf: predData.lstm ? `${predData.lstm.confidence.toFixed(1)}%` : (predData.model === 'LSTM' ? `${predData.confidence.toFixed(1)}%` : 'N/A'),
    });

    // Keep up to 30 items
    if (history.length > 30) history.pop();
    localStorage.setItem('truthscan_history', JSON.stringify(history));

    renderPredictionHistory();
}

function renderPredictionHistory() {
    const tbody = document.getElementById('tbody-prediction-history');
    const cardsContainer = document.getElementById('recent-prediction-cards');
    const history = getStoredPredictions();

    if (!tbody) return;

    if (history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No predictions recorded yet. Use the Fake News Detector tab to test articles.</td></tr>`;
        if (cardsContainer) cardsContainer.innerHTML = '';
        return;
    }

    tbody.innerHTML = history.map(item => `
        <tr>
            <td style="white-space: nowrap; color: var(--text-secondary); font-size: 0.8rem;">${item.timestamp}</td>
            <td><strong>${escapeHtml(item.snippet)}</strong></td>
            <td><span class="status-pill">${escapeHtml(item.model)}</span></td>
            <td><span style="color: ${item.result.includes('FAKE') ? 'var(--danger-red)' : 'var(--accent-green)'}; font-weight: 700;">${escapeHtml(item.result)}</span></td>
        </tr>
    `).join('');

    // Recent Expandable Cards (first 5)
    if (cardsContainer) {
        cardsContainer.innerHTML = history.slice(0, 5).map((item, idx) => `
            <div class="expander-card ${idx === 0 ? 'open' : ''}">
                <div class="expander-header" onclick="this.parentElement.classList.toggle('open')">
                    <span>[${item.timestamp}] ${escapeHtml(item.snippet)}</span>
                    <span style="color: ${item.result.includes('FAKE') ? 'var(--danger-red)' : 'var(--accent-green)'}; font-weight: 700;">${item.result.includes('FAKE') ? 'FAKE' : 'REAL'} ▾</span>
                </div>
                <div class="expander-body">
                    <div><strong>Headline / Query:</strong> ${escapeHtml(item.title || item.snippet)}</div>
                    ${item.text ? `<div style="margin-top: 0.4rem;"><strong>Article Text:</strong> ${escapeHtml(item.text.slice(0, 300))}${item.text.length > 300 ? '...' : ''}</div>` : ''}
                    <div style="margin-top: 0.6rem; display: flex; gap: 1.5rem;">
                        <span><strong>CNN Evaluation:</strong> ${item.cnn_pred} (${item.cnn_conf})</span>
                        <span><strong>LSTM Evaluation:</strong> ${item.lstm_pred} (${item.lstm_conf})</span>
                    </div>
                </div>
            </div>
        `).join('');
    }
}

async function loadSessionsData() {
    try {
        const resp = await fetch('/api/sessions');
        const data = await resp.json();

        // 1. TextCNN Training Summary
        const cnnHist = data.cnn_history;
        const cnnContainer = document.getElementById('content-cnn-history');
        if (cnnContainer) {
            if (cnnHist && cnnHist.best_val_acc) {
                cnnContainer.innerHTML = `
                    <p style="color: #FFFFFF; font-weight: 700; margin-bottom: 0.4rem;">Best Validation Accuracy: ${(cnnHist.best_val_acc * 100).toFixed(2)}%</p>
                    <ul class="params-list">
                        <li><strong>Epochs:</strong> ${cnnHist.epochs || 'N/A'}</li>
                        <li><strong>Mode:</strong> ${(cnnHist.mode || 'N/A').toUpperCase()}</li>
                        <li><strong>Compute Device:</strong> ${cnnHist.device || 'CPU'}</li>
                        <li><strong>Training Duration:</strong> ${cnnHist.total_training_time || 'N/A'}s</li>
                        <li><strong>Final Train Loss:</strong> ${cnnHist.train_loss ? cnnHist.train_loss[cnnHist.train_loss.length - 1] : 'N/A'}</li>
                        <li><strong>Final Val Loss:</strong> ${cnnHist.val_loss ? cnnHist.val_loss[cnnHist.val_loss.length - 1] : 'N/A'}</li>
                    </ul>
                `;
            } else {
                cnnContainer.textContent = 'No training history recorded yet.';
            }
        }

        // 2. LSTM Training Summary
        const lstmHist = data.lstm_history;
        const lstmContainer = document.getElementById('content-lstm-history');
        if (lstmContainer) {
            if (lstmHist && lstmHist.best_val_acc) {
                lstmContainer.innerHTML = `
                    <p style="color: #FFFFFF; font-weight: 700; margin-bottom: 0.4rem;">Best Validation Accuracy: ${(lstmHist.best_val_acc * 100).toFixed(2)}%</p>
                    <ul class="params-list">
                        <li><strong>Epochs:</strong> ${lstmHist.epochs || 'N/A'}</li>
                        <li><strong>Mode:</strong> ${(lstmHist.mode || 'N/A').toUpperCase()}</li>
                        <li><strong>Compute Device:</strong> ${lstmHist.device || 'CPU'}</li>
                        <li><strong>Training Duration:</strong> ${lstmHist.total_training_time || 'N/A'}s</li>
                        <li><strong>Final Train Loss:</strong> ${lstmHist.train_loss ? lstmHist.train_loss[lstmHist.train_loss.length - 1] : 'N/A'}</li>
                        <li><strong>Final Val Loss:</strong> ${lstmHist.val_loss ? lstmHist.val_loss[lstmHist.val_loss.length - 1] : 'N/A'}</li>
                    </ul>
                `;
            } else {
                lstmContainer.textContent = 'No training history recorded yet.';
            }
        }

        // 3. Learning Curves Chart
        const imgCurves = document.getElementById('img-sessions-curves');
        if (imgCurves && data.has_curves) {
            imgCurves.src = `${data.curves_url}?t=${Date.now()}`;
        }

    } catch (e) {
        console.warn('Could not load sessions data:', e);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
