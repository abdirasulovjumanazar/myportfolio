/* ═══════════════════════════════════════════════════════
   🏛️ DIGITAL ISLAMIC BANK — Risk Management Core
   Advanced ML (XGBoost + SHAP) + FastAPI
   ═══════════════════════════════════════════════════════ */

// ─── API Configuration (Dynamic — No hardcoded URLs) ──────────────
// API base URL is auto-detected from the current page origin.
// For local dev: window.location.origin = "http://localhost:8000"
// For production: it picks up "https://yourdankdomain.com" automatically
// Override globally by setting window.API_BASE_URL before loading this script.
const API = window.API_BASE_URL || window.location.origin;

/* ── Global Colors ── */
/* ── Global Colors: Islamic Finance Pro Palette ── */
const SVC_COLORS = {
    Murabaha: { bg: 'rgba(16, 185, 129, 0.7)', border: '#10B981' }, // Emerald
    Musharaka: { bg: 'rgba(59, 130, 246, 0.7)', border: '#3B82F6' }, // Blue
    Ijara: { bg: 'rgba(245, 158, 11, 0.7)', border: '#F59E0B' },    // Amber
    Sukuk: { bg: 'rgba(139, 92, 246, 0.7)', border: '#8B5CF6' },    // Purple
};
const SVC = Object.keys(SVC_COLORS);
const COLORS = Object.values(SVC_COLORS).map(v => v.border);

const RISK_COLORS = {
    past: { bg: 'rgba(16, 185, 129, 0.85)', border: '#10B981', text: '#10B981' },
    orta: { bg: 'rgba(245, 158, 11, 0.85)', border: '#F59E0B', text: '#F59E0B' },
    yuqori: { bg: 'rgba(239, 68, 68, 0.85)', border: '#EF4444', text: '#EF4444' },
    critical: { bg: 'rgba(153, 27, 27, 0.85)', border: '#991B1B', text: '#991B1B' },
};

const TOOLTIP_STYLE = {
    backgroundColor: 'rgba(15, 23, 42, 0.95)',
    borderColor: 'rgba(16, 185, 129, 0.3)',
    borderWidth: 1,
    titleColor: '#F8FAFC',
    bodyColor: '#94A3B8',
    padding: 12,
    cornerRadius: 10,
    titleFont: { family: "'Plus Jakarta Sans'", weight: '700', size: 14 },
    bodyFont: { family: "'Inter'", size: 12 },
    displayColors: true,
    boxWidth: 8, boxHeight: 8,
    usePointStyle: true,
};

const GRID_STYLE = {
    color: 'rgba(176, 196, 232, 0.07)',
    drawBorder: false,
    lineWidth: 1,
};

let charts = {};
let currentUser = null;

/* ── Chart defaults ─────────────────────────────── */
function updateChartDefaults() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#8A9CC4' : '#475569';
    const gridColor = isDark ? 'rgba(176, 196, 232, 0.07)' : 'rgba(0, 0, 0, 0.05)';

    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = textColor;
    Chart.defaults.animation.duration = 800;
    Chart.defaults.animation.easing = 'easeInOutCubic';
    Chart.defaults.borderColor = gridColor;
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;

    // Premium Tooltips
    Chart.defaults.plugins.tooltip = {
        ...TOOLTIP_STYLE,
        callbacks: {
            labelColor: (context) => ({
                borderColor: context.dataset.borderColor || context.dataset.backgroundColor,
                backgroundColor: context.dataset.backgroundColor,
            })
        }
    };
}
updateChartDefaults();

/* ── Plugins ────────────────────────────────────── */
const centerTextPlugin = {
    id: 'centerText',
    afterDraw(chart) {
        if (chart.canvas.id !== 'sriChart') return;
        const { ctx, chartArea: { left, top, right, bottom } } = chart;
        const cx = (left + right) / 2, cy = (top + bottom) / 2;
        ctx.save();
        ctx.textAlign = 'center';
        // Label
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        ctx.fillStyle = isDark ? '#64748B' : '#94A3B8';
        ctx.font = "600 11px 'Inter'";
        ctx.fillText("O'rtacha SRI", cx, cy - 25);
        // Big number
        const avgSRI = chart.config._avgSRI || '0.000';
        ctx.fillStyle = isDark ? '#F8FAFC' : '#0F172A';
        ctx.font = "800 32px 'Plus Jakarta Sans'";
        ctx.fillText(avgSRI, cx, cy + 10);
        // Grade
        const grade = chart.config._sriGrade || '-';
        ctx.fillStyle = '#10B981';
        ctx.font = "700 12px 'Inter'";
        ctx.letterSpacing = '1px';
        ctx.fillText(`DARAJA ${grade}`, cx, cy + 32);
        ctx.restore();
    }
};
Chart.register(centerTextPlugin);

function buildROC(auc, n = 40) {
    const pts = [{ x: 0, y: 0 }];
    for (let i = 1; i <= n; i++) {
        const x = i / n;
        const y = Math.min(1, Math.pow(x, Math.max(0.01, 1 / (auc * 3 - 1.5))));
        pts.push({ x: parseFloat(x.toFixed(3)), y: parseFloat(y.toFixed(3)) });
    }
    pts.push({ x: 1, y: 1 });
    return pts;
}

/* ── Helpers ─────────────────────────────────────── */
function getSvcColor(svc) {
    const map = {
        'Murabaha': '#C9A84C',
        'Musharaka': '#3B82F6',
        'Ijara': '#22C55E',
        'Sukuk': '#8B5CF6'
    };
    return map[svc] || '#8A9CC4';
}
function getGradient(ctx, color1, color2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2 || color1 + '33');
    return gradient;
}
function fmt(n, dec = 2) { return parseFloat(n).toFixed(dec); }
function fmtMln(n) {
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(0) + 'K';
    return n;
}
function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

async function apiFetch(url, options = {}) {
    options.credentials = 'include';
    try {
        const r = await fetch(API + url, options);
        if (r.status === 401) {
            showLogin(true);
            throw new Error("Seans tugagan, qayta kiring.");
        }
        if (!r.ok) {
            const errData = await r.json().catch(() => ({}));
            let msg = "API xatosi";
            if (typeof errData.detail === 'string') msg = errData.detail;
            else if (Array.isArray(errData.detail)) msg = errData.detail.map(e => e.msg).join(', ');
            else if (errData.detail) msg = JSON.stringify(errData.detail);
            
            throw new Error(msg);
        }
        return r.json();
    } catch (err) {
        console.error(`Fetch error (${url}):`, err);
        throw err;
    }
}

/* ── Auth Logic ──────────────────────────────────── */
function showLogin(show = true) {
    const overlay = document.getElementById('loginOverlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
        // Reset form
        if (show) {
            document.getElementById('loginUsername').value = '';
            document.getElementById('loginPassword').value = '';
            document.getElementById('loginError').style.display = 'none';
        }
    }
}

async function handleLogin() {
    const user = document.getElementById('loginUsername').value;
    const pass = document.getElementById('loginPassword').value;
    const errEl = document.getElementById('loginError');
    const btn = document.querySelector('#loginForm button');

    btn.disabled = true;
    btn.innerHTML = 'KIRILMOQDA... <i class="fas fa-spinner fa-spin"></i>';
    errEl.style.display = 'none';

    try {
        const params = new URLSearchParams();
        params.append('username', user);
        params.append('password', pass);

        const r = await fetch(API + '/api/auth/login', {
            method: 'POST',
            body: params,
            credentials: 'include'
        });

        if (!r.ok) {
            const data = await r.json();
            let msg = "Kirishda xatolik";
            if (typeof data.detail === 'string') msg = data.detail;
            else if (Array.isArray(data.detail)) msg = data.detail.map(e => e.msg).join(', ');
            else if (data.detail) msg = JSON.stringify(data.detail);
            
            throw new Error(msg);
        }

        const data = await r.json();
        console.log("Logged in:", data);
        
        await checkAuth(); // Verify and update UI
        init(); // Start model polling and dashboard load
    } catch (err) {
        errEl.textContent = err.message;
        errEl.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'KIRISH <i class="fas fa-arrow-right" style="margin-left:10px;"></i>';
    }
}

async function handleLogout() {
    try {
        await fetch(API + '/api/auth/logout', { method: 'POST', credentials: 'include' });
        location.reload(); // Refresh to clear state
    } catch (err) {
        console.error("Logout error:", err);
    }
}

async function checkAuth() {
    try {
        const user = await apiFetch('/api/auth/me');
        currentUser = user;
        showLogin(false);
        updateUIForRole(user);
        return true;
    } catch (err) {
        currentUser = null;
        showLogin(true);
        return false;
    }
}

function updateUIForRole(user) {
    const box = document.getElementById('userProfileBox');
    const nameTxt = document.getElementById('userNameText');
    const roleTxt = document.getElementById('userRoleText');
    
    if (box) box.style.display = 'block';
    if (nameTxt) nameTxt.textContent = user.username;
    if (roleTxt) roleTxt.textContent = user.role;

    // RBAC: Hide/Show nav links
    const rolePermissions = {
        'ADMIN': ['dashboard', 'predict', 'eda', 'portfolio', 'history', 'upload', 'prep', 'models', 'retrain'],
        'ANALYST': ['dashboard', 'predict', 'eda', 'portfolio', 'history'],
        'VIEWER': ['dashboard', 'eda']
    };

    const allowed = rolePermissions[user.role] || [];
    
    document.querySelectorAll('.nav-link').forEach(link => {
        const tab = link.dataset.tab;
        if (allowed.includes(tab)) {
            link.parentElement.style.display = 'block';
        } else {
            link.parentElement.style.display = 'none';
        }
    });

    // If current tab is not allowed, switch to first allowed
    const activeLink = document.querySelector('.nav-link.active');
    if (activeLink && !allowed.includes(activeLink.dataset.tab)) {
        switchTab(allowed[0] || 'dashboard');
    }
}

/* ── Tab Navigation ──────────────────────────────── */
async function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

    const targetTab = document.getElementById('tab-' + tabId);
    const targetNav = document.getElementById('nav-' + tabId);

    if (targetTab) targetTab.classList.add('active');
    if (targetNav) targetNav.classList.add('active');

    const titles = {
        dashboard: 'Executive Dashboard',
        predict: 'Kredit Tahlili & Risk Scoring',
        eda: 'Risk Metrikalari & EDA',
        portfolio: 'Portfel Optimizatsiyasi',
        history: 'Hisobotlar Tarixi',
        upload: 'Dataset Tahlili',
        prep: 'Ma\'lumotlarni Tayyorlash',
        models: 'ML Modellar Benchmark',
        retrain: 'Modelni Boshqarish'
    };
    document.getElementById('headerTitle').textContent = titles[tabId] || tabId;

    try {
        if (tabId === 'dashboard') await loadDashboard();
        if (tabId === 'eda') await loadEDA();
        if (tabId === 'history') await loadPredictionHistory();
        if (tabId === 'montecarlo') await loadMonteCarlo('Murabaha');
        if (tabId === 'stress') await loadStress();
        if (tabId === 'portfolio') await loadPortfolio();
        if (tabId === 'upload') initUpload();
        if (tabId === 'prep') initPrep();
        if (tabId === 'models') await loadModels();
        if (tabId === 'retrain') initRetrain();
    } catch (err) {
        console.error(`Tab ${tabId} yuklashda xato:`, err);
    }
}


document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        switchTab(link.dataset.tab);
    });
});

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

function updateRange(el, valId) {
    document.getElementById(valId).textContent = el.value;
}

/* ── INIT — Startup ──────────────────────────────── */
async function init() {
    initTheme();
    const loader = document.getElementById('globalLoader');
    const dot = document.getElementById('apiStatus');
    const txt = document.getElementById('apiStatusText');
    const loaderText = loader ? loader.querySelector('.loader-text') : null;

    if (loader) loader.style.display = 'flex';

    try {
        console.log("🏛️ Risk Core Initializing...");
        
        // 1. Check Auth First
        const authed = await checkAuth();
        if (!authed) {
            if (loader) loader.style.display = 'none';
            return; // Wait for login
        }

        // 2. Poll until models are ready (max 2 minutes)
        const maxAttempts = 60;
        for (let i = 0; i < maxAttempts; i++) {
            try {
                const h = await apiFetch('/api/health');
                if (h.model_trained) {
                    console.log("✅ ML Core Ready.");
                    if (loader) loader.style.display = 'none';
                    if (dot) dot.classList.add('online');
                    if (txt) txt.textContent = 'Digital Core Ready';
                    
                    // Start loading dashboard in background
                    loadDashboard().catch(e => console.error("Dashboard yuklashda xato:", e));
                    return;
                }
                if (loaderText) {
                    loaderText.innerHTML = `ML modellar o'qitilmoqda...<br><small>Hozirgi holat: <b>Ensemble training</b> (${i+1}/${maxAttempts})</small>`;
                }
            } catch (e) {
                console.warn("Backend hali tayyor emas, kutilyapti...");
            }
            if (dot) dot.classList.add('loading');
            await new Promise(res => setTimeout(res, 2000));
        }
        
        // If we reach here, it timed out
        if (loaderText) {
            loaderText.innerHTML = `⚠️ Kutish vaqti tugadi.<br><button onclick="location.reload()" class="btn-primary" style="margin-top:15px; padding:8px 20px;">Sahifani yangilash</button>`;
        }
    } catch (e) {
        console.error("Init fatal error:", e);
        if (loaderText) {
            loaderText.innerHTML = `❌ Xato: ${e.message}<br><button onclick="location.reload()" class="btn-primary" style="margin-top:15px;">Qaytadan urinish</button>`;
        }
    }
}



/* ── DASHBOARD ───────────────────────────────────── */
async function loadDashboard() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#B0C4E8' : 'rgba(15, 23, 42, 0.8)';

    const [data, perf, stressData, edaData] = await Promise.all([
        apiFetch('/api/risk-metrics'),
        apiFetch('/api/model-performance'),
        apiFetch('/api/stress-test'),
        apiFetch('/api/eda-stats')
    ]);

    // KPI Updates
    const svcs = ['Murabaha', 'Musharaka', 'Ijara', 'Sukuk'];
    const totEL = svcs.reduce((s, k) => s + (data[k]?.el_mln || 0), 0);
    const avgPD = svcs.reduce((s, k) => s + (data[k]?.avg_pd || 0), 0) / svcs.length;
    const ensAUC = perf?.Ensemble?.auc ?? perf?.Ensemble ?? 0.942;
    const sotaAUC = perf?.XGBoost?.auc ?? 0.931;

    document.getElementById('kpi-avg-pd').textContent = fmt(avgPD) + '%';
    document.getElementById('kpi-model-auc').textContent = sotaAUC.toFixed(3);
    document.getElementById('kpi-total-el').textContent = fmtMln(totEL * 1000000);
    document.getElementById('kpi-ensemble-auc').textContent = ensAUC.toFixed(3);

    // 1. VaR Chart Render (Horizontal Bar)
    destroyChart('var');
    const varCtx = document.getElementById('varChart').getContext('2d');
    charts['var'] = new Chart(varCtx, {
        type: 'bar',
        data: {
            labels: svcs,
            datasets: [{
                label: 'VaR 95%',
                data: svcs.map(s => data[s]?.var_95 ?? 0),
                backgroundColor: svcs.map(s => SVC_COLORS[s].bg),
                borderColor: svcs.map(s => SVC_COLORS[s].border),
                borderWidth: 1.5,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...TOOLTIP_STYLE,
                    callbacks: {
                        title: items => `${items[0].label}`,
                        label: ctx => [
                            ` VaR  95% = ${ctx.parsed.x.toFixed(5)}`,
                            ` CVaR 95% ≈ ${(ctx.parsed.x * 1.43).toFixed(5)}`
                        ]
                    }
                }
            },
            scales: {
                x: {
                    grid: GRID_STYLE,
                    ticks: { color: '#8A9CC4', font: { family: "'IBM Plex Mono'", size: 10 }, callback: v => v.toFixed(4) },
                    title: { display: true, text: 'VaR qiymati (kunlik)', color: '#4A5B82', font: { size: 11 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { 
                        color: textColor, // Use calculated color for visibility
                        font: { weight: '600', size: 12 },
                        autoSkip: false
                    }
                }
            }
        }
    });

    // 2. SRI Donut Render (Center Text + Dynamic Legend)
    destroyChart('sri');
    const sriCtx = document.getElementById('sriChart').getContext('2d');
    const sriDataCounts = [
        edaData?.risk_distribution?.Past ?? 0,
        edaData?.risk_distribution?.["O'rta"] ?? 0,
        edaData?.risk_distribution?.Yuqori ?? 0,
        edaData?.risk_distribution?.['Juda Yuqori'] ?? 0,
    ];
    const sriTotal = sriDataCounts.reduce((a, b) => a + b, 0);
    const avgSRI = svcs.reduce((s, k) => s + (data[k]?.sri || 0), 0) / svcs.length;
    const grade = avgSRI < 0.08 ? 'A' : (avgSRI < 0.14 ? 'B' : 'C');

    charts['sri'] = new Chart(sriCtx, {
        type: 'doughnut',
        data: {
            labels: ['Past', "O'rta", 'Yuqori', 'Juda Yuqori'],
            datasets: [{
                data: sriDataCounts,
                backgroundColor: [RISK_COLORS.past.bg, RISK_COLORS.orta.bg, RISK_COLORS.yuqori.bg, RISK_COLORS.critical.bg],
                borderColor: [RISK_COLORS.past.border, RISK_COLORS.orta.border, RISK_COLORS.yuqori.border, RISK_COLORS.critical.border],
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            cutout: '68%',
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...TOOLTIP_STYLE,
                    callbacks: {
                        label: ctx => {
                            const pct = ((ctx.parsed / sriTotal) * 100).toFixed(1);
                            return ` ${ctx.label}: ${ctx.parsed} mijoz (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
    charts['sri'].config._avgSRI = avgSRI.toFixed(3);
    charts['sri'].config._sriGrade = grade;

    const sriLegendEl = document.getElementById('sriLegend');
    const legendDefs = [
        { label: 'Past', count: sriDataCounts[0], color: '#22C55E' },
        { label: "O'rta", count: sriDataCounts[1], color: '#F59E0B' },
        { label: 'Yuqori', count: sriDataCounts[2], color: '#EF4444' },
        { label: 'Juda Yuqori', count: sriDataCounts[3], color: '#DC2626' },
    ];
    sriLegendEl.innerHTML = legendDefs.map(d => {
        const countFormatted = d.count.toLocaleString();
        const pct = ((d.count / sriTotal) * 100).toFixed(1);
        return `
            <div style="display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:8px;cursor:pointer;transition:background 0.15s;"
                 onmouseover="this.style.background='rgba(255,255,255,0.04)'"
                 onmouseout="this.style.background='transparent'">
                <span style="width:10px;height:10px;border-radius:3px;background:${d.color};flex-shrink:0;box-shadow:0 0 6px ${d.color}66"></span>
                <span style="font-size:13px;color:#F8FAFF;flex:1;font-weight:500">${d.label}</span>
                <span style="font-family:'IBM Plex Mono';font-size:12px;color:#B0C4E8;text-align:right">${countFormatted}</span>
                <span style="font-family:'IBM Plex Mono';font-size:13px;font-weight:700;color:${d.color};min-width:44px;text-align:right">${pct}%</span>
            </div>`;
    }).join('');

    // 3. ROC Curve Render (4 Model + Diagonal)
    destroyChart('roc');
    const rocCtx = document.getElementById('rocChart').getContext('2d');
    const perfData = perf || {};
    const models = [
        { name: 'Ensemble', key: 'Ensemble', color: '#F0F4FF', width: 2.5 },
        { name: 'XGBoost', key: 'XGBoost', color: '#E2B94E', width: 2 },
        { name: 'GBM', key: 'GBM', color: '#60A5FA', width: 2 },
        { name: 'Random Forest', key: 'Random Forest', color: '#2EE272', width: 2 },
    ];

    charts['roc'] = new Chart(rocCtx, {
        type: 'scatter',
        data: {
            datasets: [
                ...models.map(m => ({
                    label: `${m.name} (AUC=${(perfData[m.key]?.auc ?? 0.90).toFixed(3)})`,
                    data: buildROC(perfData[m.key]?.auc ?? 0.90),
                    showLine: true,
                    borderColor: m.color,
                    borderWidth: m.width,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: false,
                })),
                {
                    label: 'Tasodifiy model',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    showLine: true,
                    borderColor: 'rgba(255,255,255,0.18)',
                    borderDash: [6, 4],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, position: 'bottom',
                    labels: { color: '#B0C4E8', font: { family: "'Plus Jakarta Sans'", size: 12, weight: '600' }, usePointStyle: true, padding: 20 }
                },
                tooltip: {
                    ...TOOLTIP_STYLE,
                    callbacks: { label: ctx => ` FPR=${ctx.parsed.x.toFixed(3)} | TPR=${ctx.parsed.y.toFixed(3)}` }
                }
            },
            scales: {
                x: {
                    type: 'linear', min: 0, max: 1,
                    grid: GRID_STYLE,
                    title: { display: true, text: "Yolg'on musbat darajasi (FPR)", color: '#B0C4E8', font: { size: 11, weight: '600' } },
                    ticks: { color: '#B0C4E8', font: { family: "'IBM Plex Mono'", size: 10 } }
                },
                y: {
                    type: 'linear', min: 0, max: 1,
                    grid: GRID_STYLE,
                    title: { display: true, text: 'Haqiqiy musbat darajasi (TPR)', color: '#B0C4E8', font: { size: 11, weight: '600' } },
                    ticks: { color: '#B0C4E8', font: { family: "'IBM Plex Mono'", size: 10 } }
                }
            }
        }
    });

    // 4. Stress Test Render (Grouped Bar + UZS Label)
    destroyChart('stress');
    const stressCtx = document.getElementById('stressChart').getContext('2d');
    const scenarios = Object.keys(stressData);
    charts['stress'] = new Chart(stressCtx, {
        type: 'bar',
        data: {
            labels: scenarios,
            datasets: svcs.map(svc => ({
                label: svc,
                data: scenarios.map(sc => stressData[sc]?.[svc] ?? 0),
                backgroundColor: SVC_COLORS[svc].bg,
                borderColor: SVC_COLORS[svc].border,
                borderWidth: 1,
                borderRadius: 4,
            }))
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { color: '#B0C4E8', font: { size: 11 }, usePointStyle: true, padding: 16 } },
                tooltip: {
                    ...TOOLTIP_STYLE,
                    mode: 'index',
                    callbacks: {
                        title: items => `📊 ${items[0].label} ssenariy`,
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} mln UZS`
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#8A9CC4', font: { size: 10 }, maxRotation: 0 } },
                y: {
                    grid: GRID_STYLE,
                    ticks: { color: '#8A9CC4', font: { family: "'IBM Plex Mono'", size: 10 }, callback: v => v + 'M' },
                    title: { display: true, text: 'Kutilgan zarar (mln UZS)', color: '#4A5B82', font: { size: 10 } }
                }
            }
        }
    });

    // 5. Monte Carlo Render (Default: Murabaha) asenkron chaqiramiz
    loadMonteCarlo('Murabaha').catch(e => console.error(e));

    // 6. Expected vs Unexpected Loss (elChart)
    destroyChart('elChart');
    const elCtx = document.getElementById('elChart').getContext('2d');
    charts['elChart'] = new Chart(elCtx, {
        type: 'bar',
        data: {
            labels: svcs,
            datasets: [
                {
                    label: 'Expected Loss (EL)',
                    data: svcs.map(s => data[s]?.el_mln ?? 0),
                    backgroundColor: 'rgba(201,168,76,0.75)',
                    borderColor: '#C9A84C',
                    borderWidth: 1.5,
                    borderRadius: 5,
                },
                {
                    label: 'Unexpected Loss (UL)',
                    data: svcs.map(s => data[s]?.ul_mln ?? 0),
                    backgroundColor: 'rgba(139,92,246,0.75)',
                    borderColor: '#8B5CF6',
                    borderWidth: 1.5,
                    borderRadius: 5,
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: '#8A9CC4', font: { size: 11 }, usePointStyle: true, padding: 16 } },
                tooltip: TOOLTIP_STYLE
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#8A9CC4', font: { size: 11, weight: '500' } },
                    border: { color: 'rgba(176,196,232,0.1)' }
                },
                y: {
                    grid: GRID_STYLE,
                    ticks: { color: '#8A9CC4', font: { family: "'IBM Plex Mono'", size: 10 }, callback: v => v + 'M' },
                    title: { display: true, text: 'Zarar (mln UZS)', color: '#4A5B82', font: { size: 10 } },
                    border: { color: 'transparent' }
                }
            }
        }
    });

    // 7. Default Rate Portfolio
    if (edaData?.sector_default) {
        renderDefaultRateChart('defaultRateChart', edaData.sector_default);
    }

}

/* ── PREDICT FEATURE IMPORTANCE (SHAP) ── */
function renderShapChart(shapData) {
    if (!shapData) return;
    destroyChart('shap');
    const entries = Object.entries(shapData)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10);

    const labels = entries.map(([k]) =>
        k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    );
    const values = entries.map(([, v]) => parseFloat(v.toFixed(4)));
    const colors = values.map(v =>
        v >= 0 ? 'rgba(226,185,78,0.85)' : 'rgba(239,68,68,0.80)'
    );
    const borders = values.map(v => v >= 0 ? '#E2B94E' : '#EF4444');

    const ctx = document.getElementById('shapChart').getContext('2d');
    charts['shap'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'SHAP qiymati',
                data: values,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1.5,
                borderRadius: 5,
                borderSkipped: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...TOOLTIP_STYLE,
                    callbacks: {
                        label: ctx => {
                            const sign = ctx.parsed.x > 0 ? '↑ Risk oshiradi' : '↓ Risk kamaytiradi';
                            return ` ${ctx.parsed.x > 0 ? '+' : ''}${ctx.parsed.x.toFixed(4)}  (${sign})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: GRID_STYLE,
                    ticks: {
                        color: '#8A9CC4',
                        font: { family: "'IBM Plex Mono'", size: 10 },
                        callback: v => (v > 0 ? '+' : '') + v.toFixed(3)
                    },
                    title: { display: true, text: "SHAP ta'siri", color: '#4A5B82', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#F0F4FF', font: { size: 11, weight: '500' } }
                }
            }
        }
    });
}

function renderDefaultRateChart(canvasId, dataObj) {
    destroyChart(canvasId);
    const sorted = Object.entries(dataObj).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([k]) => k);
    const values = sorted.map(([, v]) => parseFloat((v * 100).toFixed(2)));

    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const range = (maxVal - minVal) || 1;

    const colors = values.map(v => {
        const norm = (v - minVal) / range; // 0 (eng yaxshi/past) dan 1 (eng yomon/yuqori) gacha
        if (norm < 0.25) return 'rgba(34, 197, 94, 0.85)';    // Yashil (Eng past risk)
        if (norm < 0.50) return 'rgba(234, 179, 8, 0.85)';    // Sariq (O'rtachadan past)
        if (norm < 0.75) return 'rgba(245, 158, 11, 0.85)';   // Zarg'aldoq (Xavfli)
        return 'rgba(239, 68, 68, 0.85)';                     // Qizil (Eng yomon)
    });

    const ctx = document.getElementById(canvasId).getContext('2d');
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Default nisbati',
                data: values,
                backgroundColor: colors,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...TOOLTIP_STYLE,
                    callbacks: { label: ctx => ` Default: ${ctx.parsed.y.toFixed(1)}%` }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#8A9CC4', font: { size: 10 }, maxRotation: 25 } },
                y: {
                    grid: GRID_STYLE,
                    ticks: { color: '#8A9CC4', font: { family: "'IBM Plex Mono'", size: 10 }, callback: v => v + '%' },
                    title: { display: true, text: 'Default nisbati (%)', color: '#4A5B82', font: { size: 10 } }
                }
            }
        }
    });
}

/**
 * Monte Carlo Simulation Chart Logic
 */
async function loadMonteCarlo(service) {
    try {
        const data = await apiFetch(`/api/monte-carlo/${service}`);
        destroyChart('monte');
        const ctx = document.getElementById('monteChart').getContext('2d');
        if (!ctx) return;

        charts['monte'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.days,
                datasets: [
                    { 
                        label: '95% Confidence', 
                        data: data.p95, 
                        borderColor: '#D4AF37', 
                        borderDash: [5, 5], 
                        fill: false, 
                        tension: 0.3,
                        pointRadius: 0
                    },
                    { 
                        label: 'Median Path', 
                        data: data.p50, 
                        borderColor: '#3B82F6', 
                        borderWidth: 2,
                        fill: false, 
                        tension: 0.4,
                        pointRadius: 0
                    },
                    { 
                        label: '5% Confidence', 
                        data: data.p5, 
                        borderColor: '#EF4444', 
                        borderDash: [5, 5], 
                        fill: false, 
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true, 
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        ...TOOLTIP_STYLE,
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#8A9CC4', font: { size: 10 } },
                        title: { display: true, text: 'Kunlar', color: '#4A5B82', font: { size: 10 } }
                    },
                    y: {
                        grid: GRID_STYLE,
                        ticks: { color: '#8A9CC4', font: { size: 10 } },
                        title: { display: true, text: 'Nizomiy Qiymat', color: '#4A5B82', font: { size: 10 } }
                    }
                }
            }
        });
    } catch (err) {
        console.error("Monte Carlo yuklashda xato:", err);
    }
}

/* ── PREDICT ─────────────────────────────────────── */
async function handlePredict() {
    const form = document.getElementById('predictForm');
    if (!form) return;
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Hisoblanmoqda... <i class="fas fa-spinner fa-spin"></i>';

    const fd = new FormData(form);
    const body = {};
    for (let [k, v] of fd.entries()) {
        body[k] = isNaN(v) || v === '' ? v : Number(v);
    }

    try {
        const result = await apiFetch('/api/predict', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        showResult(result);
        showToast('✅ Risk tahlili muvaffaqiyatli', 'success');
        refreshDashboardSilently();
    } catch (err) {
        showToast('❌ Xato: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🔮 Risk Hisoblash <i class="fas fa-shield-halved" style="margin-left:8px;"></i>';
    }
}

if (document.getElementById('predictForm')) {
    document.getElementById('predictForm').addEventListener('submit', e => {
        e.preventDefault();
        handlePredict();
    });
}

async function refreshDashboardSilently() {
    try {
        const [kpiData, riskData, edaData] = await Promise.all([
            apiFetch('/api/model-performance'),
            apiFetch('/api/risk-metrics'),
            apiFetch('/api/eda-stats')
        ]);

        // 1. KPI Update (Oltin flash animatsiyasi)
        const updateVal = (id, val) => {
            const el = document.getElementById(id);
            if (el && el.innerText !== String(val)) {
                el.innerText = val;
                el.style.transition = 'color 0.3s, text-shadow 0.3s';
                el.style.color = '#D4AF37';
                el.style.textShadow = '0 0 10px rgba(212,175,55,0.8)';
                setTimeout(() => {
                    el.style.color = '';
                    el.style.textShadow = '';
                }, 1200);
            }
        };

        const svcs = ['Murabaha', 'Musharaka', 'Ijara', 'Sukuk'];
        const totEL = svcs.reduce((s, k) => s + (riskData[k]?.el_mln || 0), 0);
        const avgPD = svcs.reduce((s, k) => s + (riskData[k]?.avg_pd || 0), 0) / svcs.length;
        const ensAUC = kpiData?.Ensemble?.auc ?? kpiData?.Ensemble ?? 0.942;
        const sotaAUC = kpiData?.XGBoost?.auc ?? 0.931;

        updateVal('kpi-avg-pd', fmt(avgPD) + '%');
        updateVal('kpi-model-auc', sotaAUC.toFixed(3));
        updateVal('kpi-total-el', fmtMln(totEL * 1000000));
        updateVal('kpi-ensemble-auc', ensAUC.toFixed(3));
        
        // Jimgina status dot signal (oltin rang yozuv)
        const dot = document.getElementById('apiStatus');
        if (dot) {
            const oldBg = dot.style.background;
            const oldBoxShadow = dot.style.boxShadow;
            dot.style.background = '#D4AF37';
            dot.style.boxShadow = '0 0 8px #D4AF37';
            setTimeout(() => {
                dot.style.background = oldBg;
                dot.style.boxShadow = oldBoxShadow;
            }, 1200);
        }

        // 2. VaR Chart Update
        if (charts['var'] && riskData) {
            charts['var'].data.datasets[0].data = svcs.map(s => riskData[s]?.var_95 ?? 0);
            charts['var'].update('active');
        }

        // 3. SRI Chart Update
        if (charts['sri'] && edaData) {
            const sriDataCounts = [
                edaData?.risk_distribution?.Past ?? 0,
                edaData?.risk_distribution?.["O'rta"] ?? 0,
                edaData?.risk_distribution?.Yuqori ?? 0,
                edaData?.risk_distribution?.['Juda Yuqori'] ?? 0,
            ];
            charts['sri'].data.datasets[0].data = sriDataCounts;
            charts['sri'].update('active');
            
            // Legend Update
            const sriTotal = sriDataCounts.reduce((a, b) => a + b, 0);
            const legendDefs = [
                { label: 'Past', count: sriDataCounts[0], color: '#22C55E' },
                { label: "O'rta", count: sriDataCounts[1], color: '#F59E0B' },
                { label: 'Yuqori', count: sriDataCounts[2], color: '#EF4444' },
                { label: 'Juda Yuqori', count: sriDataCounts[3], color: '#DC2626' },
            ];
            const sriLegendEl = document.getElementById('sriLegend');
            if (sriLegendEl) {
                sriLegendEl.innerHTML = legendDefs.map(d => {
                    const countFormatted = d.count.toLocaleString();
                    const pct = sriTotal ? ((d.count / sriTotal) * 100).toFixed(1) : 0;
                    return `
                        <div style="display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:8px;cursor:pointer;transition:background 0.15s;"
                             onmouseover="this.style.background='rgba(255,255,255,0.04)'"
                             onmouseout="this.style.background='transparent'">
                            <span style="width:10px;height:10px;border-radius:3px;background:${d.color};flex-shrink:0;box-shadow:0 0 6px ${d.color}66"></span>
                            <span style="font-size:13px;color:#F8FAFF;flex:1;font-weight:500">${d.label}</span>
                            <span style="font-family:'IBM Plex Mono';font-size:12px;color:#B0C4E8;text-align:right">${countFormatted}</span>
                            <span style="font-family:'IBM Plex Mono';font-size:13px;font-weight:700;color:${d.color};min-width:44px;text-align:right">${pct}%</span>
                        </div>`;
                }).join('');
            }
        }

        // 4. ROC Chart Update
        if (charts['roc'] && kpiData) {
            const models = [
                { name: 'Ensemble', key: 'Ensemble' },
                { name: 'XGBoost', key: 'XGBoost' },
                { name: 'GBM', key: 'GBM' },
                { name: 'Random Forest', key: 'Random Forest' },
            ];
            models.forEach((m, idx) => {
                const auc = kpiData[m.key]?.auc ?? 0.90;
                charts['roc'].data.datasets[idx].label = `${m.name} (AUC=${auc.toFixed(3)})`;
                charts['roc'].data.datasets[idx].data = buildROC(auc);
            });
            charts['roc'].update('active');
        }

    } catch (e) {
        console.error('Fon yangilanish xatosi:', e);
    }
}

function showResult(r) {
    const panel = document.getElementById('predictResult');
    if (!panel) return;
    panel.style.display = 'block';

    const resRiskCard = document.getElementById('resRiskCard');
    const resRiskText = document.getElementById('resRiskText');
    const resProbText = document.getElementById('resProbText');

    if (resRiskText) resRiskText.textContent = (r.risk_darajasi || 'UNKNOWN').toUpperCase();
    if (resProbText) resProbText.textContent = `Ehtimollik: ${(r.default_ehtimoli_pct || 0).toFixed(2)}%`;

    if (resRiskCard) {
        const colors = { 0: 'var(--risk-low)', 1: 'var(--risk-medium)', 2: 'var(--risk-high)', 3: 'var(--risk-critical)' };
        resRiskCard.style.setProperty('--accent-color', colors[r.risk_kodi || 0]);
    }

    if (r.shap_explain && typeof renderShapChart === 'function') {
        renderShapChart(r.shap_explain);
    }

    setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function getColor(p) {
    if (p < 0.10) return '#10b981';
    if (p < 0.25) return '#f59e0b';
    if (p < 0.45) return '#f97316';
    return '#ef4444';
}

function drawGauge(value) {
    const canvas = document.getElementById('gaugeChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 200, 140);
    const cx = 100, cy = 120, r = 85;
    // Background arc
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 0);
    ctx.strokeStyle = '#1a2233'; ctx.lineWidth = 14; ctx.stroke();
    // Value arc
    const angle = Math.PI + value * Math.PI;
    const grad = ctx.createLinearGradient(20, 120, 180, 120);
    grad.addColorStop(0, '#10b981'); grad.addColorStop(0.5, '#f59e0b'); grad.addColorStop(1, '#ef4444');
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, angle);
    ctx.strokeStyle = grad; ctx.lineWidth = 14; ctx.lineCap = 'round'; ctx.stroke();
}

/* ── EDA ─────────────────────────────────────────── */
async function loadEDA() {
    const data = await apiFetch('/api/eda-stats');
    // KPIs
    document.getElementById('edaKpiGrid').innerHTML = `
    <div class="kpi-card"><div class="kpi-icon"><i class="fas fa-list-check"></i></div><div class="kpi-label">Jami Yozuvlar</div><div class="kpi-value">${data.total_records.toLocaleString()}</div></div>
    <div class="kpi-card"><div class="kpi-icon"><i class="fas fa-triangle-exclamation"></i></div><div class="kpi-label">Default Nisbati</div><div class="kpi-value">${data.default_rate}%</div></div>
    <div class="kpi-card"><div class="kpi-icon"><i class="fas fa-credit-card"></i></div><div class="kpi-label">O'rt. Kredit Ball</div><div class="kpi-value">${data.avg_kredit_ball}</div></div>
    <div class="kpi-card"><div class="kpi-icon"><i class="fas fa-heart-pulse"></i></div><div class="kpi-label">O'rt. Sharia Baho</div><div class="kpi-value">${(data.avg_sharia * 100).toFixed(1)}%</div></div>
  `;

    renderDefaultRateChart('sectorChart', data.sector_default);
    renderDefaultRateChart('regionChart', data.region_default);

    // Risk distribution
    destroyChart('riskDist');
    const rd = data.risk_distribution;
    charts['riskDist'] = new Chart(document.getElementById('riskDistChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: Object.keys(rd), datasets: [{
                label: 'Soni', data: Object.values(rd),
                backgroundColor: [RISK_COLORS.past.bg, RISK_COLORS.orta.bg, RISK_COLORS.yuqori.bg, RISK_COLORS.critical.bg],
                borderColor: [RISK_COLORS.past.border, RISK_COLORS.orta.border, RISK_COLORS.yuqori.border, RISK_COLORS.critical.border],
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            plugins: { legend: { display: false }, tooltip: TOOLTIP_STYLE },
            scales: {
                x: { grid: { display: false } },
                y: { grid: GRID_STYLE, ticks: { family: "'IBM Plex Mono'", size: 10 } }
            }
        }
    });
}

/* ── MONTE CARLO ─────────────────────────────────── */
let mcChart = null;
async function loadMonteCarlo(svc, btn) {
    // Button toggle
    if (btn) {
        document.querySelectorAll('#mcBtnGroup .btn-outline').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }
    const data = await apiFetch(`/api/monte-carlo/${svc}?n_sim=3000`);
    document.getElementById('mcChartTitle').textContent = `Monte Carlo GBM — ${svc}`;
    document.getElementById('mcStats').innerHTML = `
    <div class="mc-stat"><span class="mc-stat-label">VaR 99%</span><span class="mc-stat-value">${(data.var99 * 100).toFixed(2)}%</span></div>
    <div class="mc-stat"><span class="mc-stat-label">Zarar Ehtimoli</span><span class="mc-stat-value">${(data.prob_loss * 100).toFixed(1)}%</span></div>
    <div class="mc-stat"><span class="mc-stat-label">O'rt. Final</span><span class="mc-stat-value">${(data.mean_final * 100).toFixed(1)}%</span></div>
  `;
    destroyChart('mc');
    const days = data.days;
    charts['mc'] = new Chart(document.getElementById('mcChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: days, datasets: [
                { label: 'P95 (Yuqori)', data: data.p95, borderColor: '#10b981', fill: false, pointRadius: 0, borderWidth: 2 },
                { label: 'P50 (Median)', data: data.p50, borderColor: '#3b82f6', fill: false, pointRadius: 0, borderWidth: 2.5 },
                { label: 'P5 (Quyi)', data: data.p5, borderColor: '#ef4444', fill: '+1', backgroundColor: 'rgba(239,68,68,0.05)', pointRadius: 0, borderWidth: 2 },
            ]
        },
        options: {
            plugins: { legend: { labels: { font: { size: 12 } } } },
            scales: {
                x: { title: { display: true, text: 'Kun (252 ish kuni)' } },
                y: { title: { display: true, text: 'Normallashtirilgan qiymat' }, ticks: { callback: v => (v * 100).toFixed(0) + '%' } }
            }
        }
    });
}

/* ── STRESS TEST ─────────────────────────────────── */
async function loadStress() {
    const data = await apiFetch('/api/stress-test');
    const scenarios = Object.keys(data);
    const datasets = SVC.map((svc, i) => ({
        label: svc, data: scenarios.map(s => data[s][svc]),
        backgroundColor: COLORS[i], borderRadius: 4
    }));

    destroyChart('stress');
    charts['stress'] = new Chart(document.getElementById('stressChart').getContext('2d'), {
        type: 'bar',
        data: { labels: scenarios, datasets },
        options: {
            plugins: { legend: { labels: { font: { size: 12 } } } },
            scales: { x: { stacked: false }, y: { title: { display: true, text: 'EL (mln UZS)' } } }
        }
    });

    // Table
    const container = document.getElementById('stressTableContainer');
    let html = '<table><thead><tr><th>Senariy</th>';
    SVC.forEach(s => html += `<th>${s}</th>`);
    html += '<th>JAMI</th></tr></thead><tbody>';
    scenarios.forEach(sc => {
        html += `<tr><td><strong>${sc}</strong></td>`;
        SVC.forEach(s => html += `<td>${data[sc][s]} mln</td>`);
        html += `<td><strong>${data[sc].total} mln</strong></td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

/* ── PORTFOLIO ───────────────────────────────────── */
let globalPortfolioCache = null;
async function loadPortfolio() {
    if (!globalPortfolioCache) {
        globalPortfolioCache = await apiFetch('/api/portfolio?n_portfolios=2500');
    }
    const data = globalPortfolioCache;
    destroyChart('portfolio');
    const ctx = document.getElementById('portfolioChart').getContext('2d');

    charts['portfolio'] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Efficient Frontier',
                    data: data.scatter_vol.map((v, i) => ({ x: v, y: data.scatter_ret[i], sharpe: data.scatter_sr[i] })),
                    backgroundColor: ctx => {
                        const s = ctx.raw?.sharpe || 0;
                        if (s > 1.2) return 'rgba(34, 197, 94, 0.7)';
                        if (s > 0.8) return 'rgba(201, 168, 76, 0.7)';
                        return 'rgba(239, 68, 68, 0.7)';
                    },
                    pointRadius: 4,
                    hoverRadius: 7,
                },
                {
                    label: 'Max Sharpe',
                    data: [{ x: data.max_sharpe.vol, y: data.max_sharpe.return }],
                    backgroundColor: '#F0F4FF',
                    pointStyle: 'star',
                    pointRadius: 10,
                    borderWidth: 2,
                    borderColor: '#C9A84C'
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: '#8A9CC4', font: { family: "'Plus Jakarta Sans'" } } },
                tooltip: {
                    ...TOOLTIP_STYLE,
                    callbacks: {
                        label: ctx => ` Risq: ${ctx.raw.x.toFixed(2)}%, Daromad: ${ctx.raw.y.toFixed(2)}%, Sharpe: ${ctx.raw.sharpe?.toFixed(2) || 'N/A'}`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Volatillik (Risk %)', color: '#4A5B82', font: { size: 11 } },
                    grid: GRID_STYLE,
                    ticks: { color: '#8A9CC4', font: { family: "'IBM Plex Mono'", size: 10 } }
                },
                y: {
                    title: { display: true, text: 'Kutilgan Daromad (%)', color: '#4A5B82', font: { size: 11 } },
                    grid: GRID_STYLE,
                    ticks: { color: '#8A9CC4', font: { family: "'IBM Plex Mono'", size: 10 } }
                }
            }
        }
    });

    // Comparison cards
    const comp = document.getElementById('portfolioComparison');
    if (!comp) return;
    const portfels = [
        { title: '🏦 Hozirgi Portfel', d: data.current, accent: '#8A9CC4' },
        { title: '⭐ Optimal (Max Sharpe)', d: data.max_sharpe, accent: '#C9A84C' },
        { title: '🔵 Konservativ (Min Risk)', d: data.min_risk, accent: '#3B82F6' },
    ];
    comp.className = 'kpi-grid fade-up';
    comp.innerHTML = portfels.map(p => `
    <div class="kpi-card" style="--accent-color: ${p.accent}">
      <div class="kpi-header">
        <span class="kpi-label">${p.title}</span>
        <i class="fas fa-layer-group kpi-icon"></i>
      </div>
      <div class="kpi-value" style="font-size:24px">${p.d.return}% / ${p.d.vol}%</div>
      <div class="kpi-sub">Sharpe Ratio: ${p.d.sharpe?.toFixed(2) || 'N/A'}</div>
      <div style="margin-top:16px;">
        ${SVC.map(s => {
        const w = (p.d.weights || {})[s] || 0;
        return `
          <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
            <span style="color:#8A9CC4">${s}</span>
            <span style="color:#F0F4FF; font-family:'IBM Plex Mono'">${(w * 100).toFixed(0)}%</span>
          </div>
          <div style="height:4px; background:rgba(255,255,255,0.05); border-radius:2px; margin-bottom:8px; overflow:hidden;">
            <div style="height:100%; width:${w * 100}%; background:${SVC_COLORS[s].border}"></div>
          </div>`;
    }).join('')}
      </div>
    </div>`).join('');
}

/* ── MODELS ──────────────────────────────────────── */
async function loadModels() {
    const data = await apiFetch('/api/model-performance');
    // Using actual backend models but ensuring specific order for aesthetics
    const available = Object.keys(data);
    const names = ['Ensemble', 'XGBoost', 'GBM', 'Random Forest'].filter(n => available.includes(n));
    if (names.length < 4 && available.includes('Logistic Regression')) names.push('Logistic Regression');

    // KPI grid
    document.getElementById('modelKpiGrid').innerHTML = names.map((n, i) => `
    <div class="kpi-card" style="--accent-color: ${i === 0 ? '#F0F4FF' : (i === 1 ? '#C9A84C' : (i === 2 ? '#3B82F6' : '#22C55E'))}">
      <div class="kpi-header">
        <span class="kpi-label">${n} Model</span>
        <i class="fas fa-robot kpi-icon"></i>
      </div>
      <div class="kpi-value">${(data[n]?.auc ?? 0).toFixed(3)}</div>
      <div class="kpi-sub">AUC Metrikasi | F1: ${(data[n]?.f1 ?? 0).toFixed(3)}</div>
    </div>`).join('');

    // ROC chart on Models tab
    destroyChart('rocModels');
    const rocCtx = document.getElementById('rocChartModels');
    if (rocCtx) {
        charts['rocModels'] = new Chart(rocCtx.getContext('2d'), {
            type: 'scatter',
            data: {
                datasets: [
                    ...names.map((n, i) => ({
                        label: `${n} (AUC=${(data[n]?.auc ?? 0).toFixed(3)})`,
                        data: (data[n]?.roc_fpr || []).map((fpr, j) => ({ x: fpr, y: data[n].roc_tpr[j] })),
                        showLine: true,
                        borderColor: i === 0 ? '#F0F4FF' : (i === 1 ? '#E2B94E' : (i === 2 ? '#3B82F6' : '#2EE272')),
                        borderWidth: i === 0 ? 2.5 : 2,
                        pointRadius: 0,
                        tension: 0.4,
                        fill: false,
                    })),
                    {
                        label: 'Tasodifiy',
                        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                        showLine: true,
                        borderColor: 'rgba(255,255,255,0.15)',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom', labels: { color: '#8A9CC4', font: { size: 11 }, usePointStyle: true } },
                    tooltip: {
                        ...TOOLTIP_STYLE,
                        callbacks: { label: ctx => ` FPR: ${ctx.parsed.x.toFixed(3)} | TPR: ${ctx.parsed.y.toFixed(3)}` }
                    }
                },
                scales: {
                    x: { type: 'linear', min: 0, max: 1, grid: GRID_STYLE, title: { display: true, text: 'FPR', color: '#4A5B82' } },
                    y: { type: 'linear', min: 0, max: 1, grid: GRID_STYLE, title: { display: true, text: 'TPR', color: '#4A5B82' } }
                }
            }
        });
    }

    // Bar chart
    destroyChart('modelBar');
    const barCtx = document.getElementById('modelBarChart');
    if (barCtx) {
        charts['modelBar'] = new Chart(barCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: names,
                datasets: [
                    { label: 'AUC', data: names.map(n => data[n]?.auc ?? 0), backgroundColor: 'rgba(201,168,76,0.85)', borderRadius: 4 },
                    { label: 'F1-Score', data: names.map(n => data[n]?.f1 ?? 0), backgroundColor: 'rgba(59,130,246,0.85)', borderRadius: 4 },
                    { label: 'Accuracy', data: names.map(n => data[n]?.accuracy ?? 0), backgroundColor: 'rgba(34,197,94,0.85)', borderRadius: 4 },
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, labels: { color: '#8A9CC4', font: { size: 11 } } },
                    tooltip: TOOLTIP_STYLE
                },
                scales: {
                    y: { min: 0.5, max: 1.0, grid: GRID_STYLE, ticks: { color: '#8A9CC4' } },
                    x: { grid: { display: false }, ticks: { color: '#F0F4FF', font: { weight: '600' } } }
                }
            }
        });
    }

    // Frontier Chart in Models Tab (Shared with Portfolio logic but maybe different presentation)
    try {
        if (!globalPortfolioCache) {
            globalPortfolioCache = await apiFetch('/api/portfolio?n_portfolios=2500');
        }
        const portData = globalPortfolioCache;
        destroyChart('frontier');
        const fCtx = document.getElementById('frontierChart');
        if (fCtx) {
            charts['frontier'] = new Chart(fCtx.getContext('2d'), {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Efficient Frontier',
                        data: portData.scatter_vol.map((v, i) => ({ x: v, y: portData.scatter_ret[i], sharpe: portData.scatter_sr[i] })),
                        backgroundColor: ctx => {
                            const s = ctx.raw?.sharpe || 0;
                            return s > 1.2 ? 'rgba(34, 197, 94, 0.6)' : (s > 0.8 ? 'rgba(201, 168, 76, 0.6)' : 'rgba(239, 68, 68, 0.6)');
                        },
                        pointRadius: 2.5
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: { ...TOOLTIP_STYLE, callbacks: { label: ctx => ` Risk: ${ctx.raw.x.toFixed(2)}%, Return: ${ctx.raw.y.toFixed(2)}%` } }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Annualized Volatility (%)' }, grid: GRID_STYLE },
                        y: { title: { display: true, text: 'Annualized Return (%)' }, grid: GRID_STYLE }
                    }
                }
            });
        }
    } catch (e) { console.error("Frontier chart error:", e); }
}

/* ── DATA PREP (DEDICATED) ─────────────────────── */
function initPrep() {
    const ds = document.getElementById('prepDropZone');
    const fi = document.getElementById('prepFileInput');
    if (!ds || !fi) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
        ds.addEventListener(e, x => { x.preventDefault(); x.stopPropagation(); });
    });

    ds.addEventListener('dragover', () => ds.style.borderColor = 'var(--accent1)');
    ds.addEventListener('dragleave', () => ds.style.borderColor = 'var(--border)');

    ds.addEventListener('drop', e => {
        ds.style.borderColor = 'var(--border)';
        const file = e.dataTransfer.files[0];
        if (file) handlePrepUpload(file);
    });

    fi.onchange = () => { if (fi.files[0]) handlePrepUpload(fi.files[0]); };
}

/* ── SMART RETRAIN (PostgreSQL + SHA-256) ────────── */
async function loadTrainingInfo() {
    try {
        const stats = await apiFetch('/api/training-stats');
        const elTotal = document.getElementById('db-total-rows');
        if (elTotal) elTotal.textContent = (stats.db_row_count || 0).toLocaleString();
        
        const elVer = document.getElementById('model-version-text');
        if (elVer) elVer.textContent = stats.model_name || '—';
        
        const elAuc = document.getElementById('last-auc-text');
        if (elAuc) elAuc.textContent = (stats.ensemble_auc || stats.best_auc || 0).toFixed(4);

        // Agar pipeline hozir ishlayotgan bo'lsa, pollingni boshlaymiz
        if (stats.pipeline_running) {
            console.log("Pipeline already running, starting polling...");
            pollPipelineStatus();
        }
    } catch (err) {
        console.error("Training stats yuklashda xato:", err);
    }
}

/* ── RETRAIN (ONLINE LEARNING) ─────────────────── */
function initRetrain() {
    loadTrainingInfo(); // Tab ochilganda joriy statistika yuklanadi
    
    const ds = document.getElementById('retrainDropZone');
    const fi = document.getElementById('retrainFileInput');
    if (!ds || !fi) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
        ds.addEventListener(e, x => { x.preventDefault(); x.stopPropagation(); });
    });

    ds.addEventListener('dragover', () => ds.style.borderColor = 'var(--brand-primary)');
    ds.addEventListener('dragleave', () => ds.style.borderColor = 'var(--border-default)');

    ds.addEventListener('drop', e => {
        ds.style.borderColor = 'var(--border-default)';
        const file = e.dataTransfer.files[0];
        if (file) handleRetrainUpload(file);
    });

    fi.onchange = () => { if (fi.files[0]) handleRetrainUpload(fi.files[0]); };
}

async function handleRetrainUpload(file) {
    if (!file) return;
    
    const status = document.getElementById('retrainStatus');
    const progress = document.getElementById('retrainProgress');
    const resultBox = document.getElementById('retrainResult');
    
    status.style.display = 'block';
    status.innerHTML = `⏳ Fayl yuklanmoqda: <b>${file.name}</b>...`;
    progress.style.display = 'block';
    resultBox.style.display = 'none';

    const fd = new FormData();
    fd.append('file', file);

    try {
        const res = await apiFetch('/api/retrain', {
            method: 'POST',
            body: fd
        });

        if (res.status === 'started') {
            status.innerHTML = `<span style="color:var(--risk-low)"><i class="fas fa-check-circle"></i> Fayl qo'shildi (${res.added_rows} yangi, ${res.duplicate_rows} dublikat). Modelni o'qitish boshlandi...</span>`;
            pollPipelineStatus();
        } else {
            throw new Error(res.detail || "Noma'lum xato");
        }
    } catch (err) {
        status.innerHTML = `<span style="color:var(--risk-high)"><i class="fas fa-exclamation-triangle"></i> Xato: ${err.message}</span>`;
        if (progress) progress.style.display = 'none';
    }
}

async function handleRetrainManual() {
    const btn = document.getElementById('startRetrainBtn');
    const loader = document.getElementById('retrainProgress');
    const statusBox = document.getElementById('retrainStatus');
    
    if (!confirm("Bazada mavjud barcha ma'lumotlar asosida modelni qayta o'qitmoqchimisiz?")) return;
    
    if (btn) btn.disabled = true;
    if (loader) loader.style.display = 'block';
    
    try {
        const result = await apiFetch('/api/retrain', { method: 'POST' });
        
        if (result.status === 'started') {
            console.log("Retrain started in background...");
            pollPipelineStatus();
        } else {
            throw new Error(result.detail || "Trigger xatosi");
        }
    } catch (err) {
        console.error("Retrain error:", err);
        if (statusBox) {
            statusBox.style.display = 'block';
            statusBox.innerHTML = `<span style="color:var(--risk-high)"><i class="fas fa-exclamation-triangle"></i> Xato: ${err.message}</span>`;
        } else {
            alert("Retrain xatosi: " + err.message);
        }
        if (btn) btn.disabled = false;
        if (loader) loader.style.display = 'none';
    }
}

async function pollPipelineStatus() {
    const btn = document.getElementById('startRetrainBtn');
    const loader = document.getElementById('retrainProgress');
    const statusText = loader.querySelector('p');
    const statusBox = document.getElementById('retrainStatus');

    try {
        const res = await apiFetch('/api/pipeline-status');
        if (res.running) {
            if (statusText) statusText.textContent = "Model orqa fonda o'qitilmoqda... (Check API logs)";
            setTimeout(pollPipelineStatus, 3000); 
        } else {
            if (btn) btn.disabled = false;
            if (loader) loader.style.display = 'none';
            
            if (res.last_error) {
                if (statusBox) statusBox.innerHTML = `<span style="color:var(--risk-high)">❌ Xato: ${res.last_error}</span>`;
            } else {
                if (statusBox) statusBox.innerHTML = `<span style="color:var(--risk-low)">✅ Model muvaffaqiyatli qayta o'qitildi! Yangi AUC: ${res.auc.toFixed(4)}</span>`;
                loadTrainingInfo();
                loadDashboard();
            }
        }
    } catch (e) {
        console.error("Polling error:", e);
        setTimeout(pollPipelineStatus, 5000);
    }
}

async function handlePrepUpload(file) {
    const st = document.getElementById('prepStatus');
    const resDiv = document.getElementById('prepResult');
    if (!st || !resDiv) return;

    st.style.display = 'block';
    st.className = 'upload-status info';
    st.textContent = `⏳ ${file.name} tayyorlanmoqda...`;
    resDiv.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const data = await fetch(API + '/api/upload-dataset', {
            method: 'POST',
            body: formData,
            credentials: 'include'
        }).then(r => r.json());

        if (data.error) throw new Error(data.error);

        st.className = 'upload-status success';
        st.textContent = `✅ Dataset tayyor!`;
        resDiv.style.display = 'block';

        const dlBtn = document.getElementById('finalPrepDownloadBtn');
        if (data.prepared_data_ready && data.prepared_data_b64) {
            dlBtn.onclick = () => {
                const link = document.createElement('a');
                link.href = 'data:text/csv;base64,' + data.prepared_data_b64;
                link.download = `tayyorlangan_${file.name.split('.')[0]}.csv`;
                link.click();
            };
        } else {
            throw new Error("Tayyorlangan ma'lumotlarni qayta ishlashda xato yuz berdi.");
        }
    } catch (e) {
        st.className = 'upload-status error';
        st.textContent = `❌ Xato: ${e.message}`;
    }
}

/* ── AUTO ANALYZER / UPLOAD ─────────────────────── */
function initUpload() {
    const ds = document.getElementById('dropZone');
    const fi = document.getElementById('fileInput');
    const st = document.getElementById('uploadStatus');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
        ds.addEventListener(e, x => { x.preventDefault(); x.stopPropagation(); });
    });

    ds.addEventListener('dragover', () => ds.style.borderColor = 'var(--accent1)');
    ds.addEventListener('dragleave', () => ds.style.borderColor = 'var(--border)');

    ds.addEventListener('drop', e => {
        ds.style.borderColor = 'var(--border)';
        const file = e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    });

    fi.onchange = () => { if (fi.files[0]) handleFileUpload(fi.files[0]); };
}

async function handleFileUpload(file) {
    const st = document.getElementById('uploadStatus');
    const resDiv = document.getElementById('autoAnalysisResults');
    st.style.display = 'block';
    st.className = 'upload-status info';
    st.textContent = `⏳ ${file.name} tahlil qilinmoqda (bu 10-20 soniya olishi mumkin)...`;
    resDiv.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const data = await fetch(API + '/api/upload-dataset', {
            method: 'POST',
            body: formData,
            credentials: 'include'
        }).then(r => r.json());

        if (data.error) throw new Error(data.error);

        st.className = 'upload-status success';
        st.textContent = `✅ Tahlil yakunlandi: ${file.name}`;
        renderAutoAnalysis(data, file.name);
    } catch (e) {
        st.className = 'upload-status error';
        st.textContent = `❌ Xato: ${e.message}`;
    }
}

function renderAutoAnalysis(data, filename) {
    const resDiv = document.getElementById('autoAnalysisResults');
    resDiv.style.display = 'block';

    // KPIs
    document.getElementById('autoKpiRows').innerHTML = `
    <div class="kpi-label">Dataset Hajmi</div>
    <div class="kpi-value">${data.basic.rows.toLocaleString()}</div>
    <div class="kpi-sub">${data.basic.cols} ta ustun | ${data.basic.missing_pct[Object.keys(data.basic.missing_pct)[0]]}% bo'sh</div>
  `;

    if (data.risk) {
        document.getElementById('autoKpiRisk').innerHTML = `
      <div class="kpi-label">Default Nisbati (Target: ${data.detected_columns.target})</div>
      <div class="kpi-value">${data.risk.default_rate}%</div>
      <div class="kpi-sub">${data.risk.default_count} ta default holati topildi</div>
    `;
    } else {
        document.getElementById('autoKpiRisk').innerHTML = `<p style="color:var(--text-muted); font-size:13px">Target (default) ustuni aniqlanmadi. Risk tahlili to'liq emas.</p>`;
    }

    // Categorical Analysis
    const catDiv = document.getElementById('autoCatAnalyzers');
    catDiv.innerHTML = '';
    if (data.risk && data.risk.by_category) {
        for (let [col, vals] of Object.entries(data.risk.by_category)) {
            let html = `<div class="auto-cat-row"><div class="auto-cat-title">${col}</div>`;
            for (let [k, v] of Object.entries(vals)) {
                html += `
          <div class="prob-row">
            <span class="prob-name" style="width:100px">${k}</span>
            <div class="prob-bar-bg"><div class="prob-bar" style="width:${v}%; background:${getColor(v / 100)}"></div></div>
            <span class="prob-pct">${v}%</span>
          </div>`;
            }
            html += '</div>';
            catDiv.innerHTML += html;
        }
    } else {
        catDiv.innerHTML = '<p style="color:var(--text-muted)">Kategorik tahlil uchun mos ustunlar topilmadi.</p>';
    }

    // Importance Chart
    destroyChart('autoImp');
    if (data.ml && data.ml.feature_importance) {
        const fi = data.ml.feature_importance;
        charts['autoImp'] = new Chart(document.getElementById('autoImportanceChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: fi.map(x => x[0]),
                datasets: [{ label: 'Muhimlik', data: fi.map(x => x[1]), backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 4 }]
            },
            options: { indexAxis: 'y', plugins: { legend: { display: false } } }
        });
    }

    // ROC Chart
    destroyChart('autoRoc');
    if (data.ml && data.ml.roc) {
        charts['autoRoc'] = new Chart(document.getElementById('autoRocChart').getContext('2d'), {
            type: 'line',
            data: {
                datasets: [
                    { label: 'Model ROC', data: data.ml.roc.fpr.map((f, i) => ({ x: f, y: data.ml.roc.tpr[i] })), borderColor: 'var(--accent1)', fill: false, pointRadius: 0, borderWidth: 3 },
                    { label: 'Random', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: 'var(--text-muted)', borderDash: [5, 5], pointRadius: 0, borderWidth: 1 }
                ]
            },
            options: { scales: { x: { type: 'linear', min: 0, max: 1 }, y: { min: 0, max: 1 } } }
        });
    }

    // Summary
    const sumBox = document.getElementById('autoSummaryBox');
    if (data.summary) {
        sumBox.innerHTML = `<strong>📋 Xulosa va Tavsiya:</strong><p>${data.summary.recommendation}</p>`;
        sumBox.style.display = 'block';
    } else {
        sumBox.style.display = 'none';
    }

    resDiv.scrollIntoView({ behavior: 'smooth' });

    // Prepared Data Download
    const prepBox = document.getElementById('preparedDataBox');
    const dlBtn = document.getElementById('downloadPreparedBtn');
    if (data.prepared_data_ready && data.prepared_data_b64) {
        prepBox.style.display = 'block';
        dlBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = 'data:text/csv;base64,' + data.prepared_data_b64;
            link.download = `tayyor_${filename || 'dataset'}.csv`;
            link.click();
        };
    } else {
        if (prepBox) prepBox.style.display = 'none';
    }
}

/* ── HISTORY ─────────────────────────────────────── */
async function loadPredictionHistory() {
    const tbody = document.getElementById('historyBody');
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">⌛ Yuklanmoqda...</td></tr>';

    try {
        const data = await apiFetch('/api/history');
        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">Hozircha ma\'lumotlar yo\'q.</td></tr>';
            return;
        }

        tbody.innerHTML = data.map(r => {
            let badgeClass = 'success';
            const rd = (r.risk_darajasi || '').toLowerCase();
            if (rd.includes('yuqori')) badgeClass = 'danger';
            else if (rd.includes("o'rta")) badgeClass = 'warning';
            else if (rd.includes('unknown')) badgeClass = 'unknown';

            return `
            <tr>
                <td>${r.timestamp}</td>
                <td>${r.xizmat_turi}</td>
                <td>${r.mintaqa}</td>
                <td>${fmtMln(r.moliyalash_miqdori)} UZS</td>
                <td>${fmt(r.pd_qiymati * 100)}%</td>
                <td><span class="badge ${badgeClass}">${r.risk_darajasi}</span></td>
                <td>${fmt(r.sri_indeksi, 4)}</td>
            </tr>`;
        }).join('');
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#ef4444;">❌ Xato: ${err.message}</td></tr>`;
    }
}

/* ── START ───────────────────────────────────────── */
window.handleRetrainManual = handleRetrainManual;

function initTheme() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;
    const html = document.documentElement;
    const icon = themeToggle.querySelector('i');

    const savedTheme = localStorage.getItem('theme') || 'dark';
    html.setAttribute('data-theme', savedTheme);
    if (icon) updateThemeIcon(savedTheme, icon);

    themeToggle.addEventListener('click', () => {
        const current = html.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        if (icon) updateThemeIcon(next, icon);
        updateChartDefaults();
        const activeLink = document.querySelector('.nav-link.active');
        if (activeLink) {
            switchTab(activeLink.dataset.tab);
        }
    });
}

function updateThemeIcon(theme, icon) {
    if (!icon) return;
    if (theme === 'light') {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
}

function showToast(message, type = 'info') {
    let toast = document.getElementById('globalToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'globalToast';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.className = `global-toast toast-${type}`;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 4000);
}

window.handlePredict = handlePredict;
window.showToast = showToast;
window.switchTab = switchTab;
window.toggleSidebar = toggleSidebar;
window.handleLogin = handleLogin;
window.handleLogout = handleLogout;
window.updateRange = updateRange;

// Global HTML helpers
window.toggleSidebar = function() {
    document.getElementById('sidebar').classList.toggle('open');
};

// Start application
init();

