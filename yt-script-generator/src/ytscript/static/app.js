// Global State
let activeTab = 'generator';
let pollingInterval = null;
let currentActiveTaskId = null;
let currentLoadedScriptText = "";
let currentLoadedReportText = "";

// Recommended topics per niche
const NICHES_SUGGESTIONS = {
    tech: [
        "Is this $1000 gadget actually worth it, or are you just paying for the logo?",
        "5 Secrets of the YouTube Algorithm Nobody Tells You",
        "I Tried Coding in 2026 for 30 Days (Crazy Results)"
    ],
    finance: [
        "How to start a business in 2026 (Complete Step-by-Step Guide)",
        "Why You're Failing at Investing (And How to Fix It)",
        "The Ultimate Guide to Passive Income in 2026"
    ],
    health: [
        "5 Daily Habits That Are Secretly Ruining Your Sleep",
        "I Drank 3 Liters of Water for 30 Days (Real Body Results)",
        "The Truth About Keto Diet Nobody Tells You"
    ],
    true_crime: [
        "The Mysterious Disappearance of [Name]: What Really Happened?",
        "5 Unsolved Cases That Will Keep You Up at Night",
        "The Truth Behind the Area 51 Incident"
    ],
    history: [
        "5 Shocking Historical Facts They Don't Teach You in School",
        "How One Small Mistake Changed the Course of WW2 Forever",
        "The Secret Private Life of Napoleon Bonaparte"
    ],
    comedy: [
        "Why Modern Dating is Actually Hilarious",
        "10 Things Everyone Does But Nobody Admits",
        "If Tech Support Was 100% Honest"
    ]
};

function populateNicheSuggestions() {
    const nicheSelect = document.getElementById("niche");
    const suggestionsList = document.getElementById("suggestions-list");
    if (!nicheSelect || !suggestionsList) return;

    const selectedNiche = nicheSelect.value;
    const suggestions = NICHES_SUGGESTIONS[selectedNiche] || [];

    suggestionsList.innerHTML = "";
    suggestions.forEach(topicText => {
        const chip = document.createElement("div");
        chip.className = "suggestion-chip";
        chip.textContent = topicText;
        chip.title = "Click to select this topic";
        
        chip.addEventListener("click", () => {
            document.getElementById("topic").value = topicText;
        });

        suggestionsList.appendChild(chip);
    });
}

// Recommended series templates per niche
const SERIES_SUGGESTIONS = {
    tech: [
        {
            title: "Mastering Python in 3 Steps",
            arc: "A 3-part video series teaching python basics: Episode 1 covers variables and basic syntax; Episode 2 covers control flow and loops; Episode 3 covers functions and building a calculator."
        },
        {
            title: "How to Build a SaaS in 2026",
            arc: "A 4-part series from scratch to launch. Ep 1: Architecture & database; Ep 2: Building the FastAPI backend; Ep 3: Designing the frontend; Ep 4: Stripe billing and deploying to cloud."
        }
    ],
    finance: [
        {
            title: "The Passive Income Blueprint",
            arc: "A 3-part guide to building cash flow. Ep 1: High-yield savings & dividend stocks; Ep 2: Launching a digital product; Ep 3: Automating operations with AI."
        },
        {
            title: "How to Start Investing in Your 20s",
            arc: "A 3-episode crash course on wealth-building. Ep 1: Compound interest basics; Ep 2: Index funds & ETFs; Ep 3: Portfolio allocation and risk tolerance."
        }
    ],
    health: [
        {
            title: "The 30-Day Energy Reset",
            arc: "A 3-part series on optimizing daily physical performance. Ep 1: Mastering sleep hygiene; Ep 2: Simple hydration and nutrition hacks; Ep 3: Setting up a 15-minute home workout routine."
        }
    ],
    true_crime: [
        {
            title: "Unsolved Secrets of the Deep Ocean",
            arc: "A 3-episode documentary series. Ep 1: The vanishing of the Mary Celeste; Ep 2: The mysterious sounds of the bloop; Ep 3: Deep sea anomalies and lost civilizations."
        }
    ],
    history: [
        {
            title: "How Rome Really Fell",
            arc: "A 3-part historical breakdown of the Roman empire's decline. Ep 1: Economic instability and hyperinflation; Ep 2: The barbarian invasions; Ep 3: Political corruption and split of empires."
        }
    ],
    comedy: [
        {
            title: "If Tech Giants Were People",
            arc: "A 3-episode comedy sketch series. Ep 1: The dinner party confrontation; Ep 2: The job interview nightmare; Ep 3: Planning a vacation together."
        }
    ]
};

function populateSeriesSuggestions() {
    const nicheSelect = document.getElementById("series-niche");
    const suggestionsList = document.getElementById("series-suggestions-list");
    if (!nicheSelect || !suggestionsList) return;

    const selectedNiche = nicheSelect.value;
    const suggestions = SERIES_SUGGESTIONS[selectedNiche] || [];

    suggestionsList.innerHTML = "";
    suggestions.forEach(item => {
        const chip = document.createElement("div");
        chip.className = "suggestion-chip";
        chip.textContent = item.title;
        chip.title = "Click to load this series template";
        
        chip.addEventListener("click", () => {
            document.getElementById("series-title").value = item.title;
            document.getElementById("series_arc").value = item.arc;
        });

        suggestionsList.appendChild(chip);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initForms();
    initSeriesForm();
    initPlannerForm();
    loadConfigSummary();
    
    // Bind Library Tab button click
    document.getElementById("nav-history-btn").addEventListener("click", loadHistory);

    // Initialize Niche Suggestions
    const nicheSelect = document.getElementById("niche");
    if (nicheSelect) {
        nicheSelect.addEventListener("change", populateNicheSuggestions);
        populateNicheSuggestions();
    }

    // Initialize Series Suggestions
    const seriesNicheSelect = document.getElementById("series-niche");
    if (seriesNicheSelect) {
        seriesNicheSelect.addEventListener("change", populateSeriesSuggestions);
        populateSeriesSuggestions();
    }
});

// 1. Navigation Tab Controller
function initTabs() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            
            navItems.forEach(nav => nav.classList.remove("active"));
            tabContents.forEach(content => content.classList.remove("active"));

            item.classList.add("active");
            document.getElementById(`tab-${targetTab}`).classList.add("active");
            
            activeTab = targetTab;
            updateHeaderTitles();
        });
    });
}

function updateHeaderTitles() {
    const titleEl = document.getElementById("page-title");
    const subtitleEl = document.getElementById("page-subtitle");

    if (activeTab === 'generator') {
        titleEl.textContent = "Script Generator";
        subtitleEl.textContent = "Configure and run the multi-stage script generation pipeline";
    } else if (activeTab === 'history') {
        titleEl.textContent = "Script Library";
        subtitleEl.textContent = "Browse, preview, and manage your previously generated scripts";
    } else if (activeTab === 'settings') {
        titleEl.textContent = "API Configurations";
        subtitleEl.textContent = "Manage API credentials and load-balancing rotation parameters";
    } else if (activeTab === 'series') {
        titleEl.textContent = "Series Studio";
        subtitleEl.textContent = "Design show outlines, sequential scripts, and continuity arcs";
    } else if (activeTab === 'planner') {
        titleEl.textContent = "Channel Planner";
        subtitleEl.textContent = "Perform video audits, transcription reviews, and content gap scheduling";
    }
}

// Toggle Advanced Form Section
function toggleAdvancedOptions() {
    const section = document.getElementById("advanced-section");
    const chevron = document.getElementById("advanced-chevron");
    
    if (section.classList.contains("hidden")) {
        section.classList.remove("hidden");
        chevron.style.transform = "rotate(180deg)";
    } else {
        section.classList.add("hidden");
        chevron.style.transform = "rotate(0deg)";
    }
}

// 2. Load API Configurations
async function loadConfigSummary() {
    try {
        const response = await fetch("/api/config");
        const config = await response.json();
        
        // Update top-bar count indicators
        document.getElementById("gemini-key-count").textContent = config.gemini_key_count;
        document.getElementById("openai-key-count").textContent = config.openai_key_count;
        
        // Update settings forms if in settings panel
        document.getElementById("settings-gemini-key-count").textContent = config.gemini_key_count;
        document.getElementById("settings-openai-key-count").textContent = config.openai_key_count;
    } catch (err) {
        console.error("Failed to load configuration summary:", err);
    }
}

// 3. Form Submissions
function initForms() {
    // A. Generation Form Submission
    const genForm = document.getElementById("generator-form");
    genForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const formData = new FormData(genForm);
        
        // Parse raw textarea split lines
        const refsRaw = formData.get("references") || "";
        const references = refsRaw.split("\n").map(l => l.trim()).filter(l => l !== "");
        
        const payload = {
            topic: formData.get("topic"),
            niche: formData.get("niche"),
            tone: formData.get("tone"),
            duration: parseFloat(formData.get("duration")),
            audience: formData.get("audience"),
            provider: formData.get("provider"),
            model: formData.get("model") || null,
            voice_sample: formData.get("voice_sample") || null,
            references: references
        };

        // Reset and show progress tracker card
        resetProgressStepper();
        document.getElementById("placeholder-card").classList.add("hidden");
        document.getElementById("results-preview-section").classList.add("hidden");
        document.getElementById("progress-card").classList.remove("hidden");
        
        setSubmitButtonState(true);

        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            currentActiveTaskId = data.task_id;
            startPollingTaskStatus(data.task_id);
        } catch (err) {
            console.error("Generation request failed:", err);
            updateConsoleLog("Error starting pipeline: " + err.message, "text-red");
            setSubmitButtonState(false);
        }
    });

    // B. Settings Form Submission
    const settingsForm = document.getElementById("settings-form");
    settingsForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const payload = {
            gemini_keys: document.getElementById("gemini_keys").value,
            openai_keys: document.getElementById("openai_keys").value
        };

        try {
            const response = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const res = await response.json();
            
            alert(res.message);
            loadConfigSummary();
        } catch (err) {
            alert("Error saving API keys: " + err.message);
        }
    });
}

function setSubmitButtonState(disabled) {
    const btn = document.getElementById("submit-btn");
    if (disabled) {
        btn.disabled = true;
        btn.innerHTML = `<i class="spin" data-lucide="refresh-cw"></i> Pipeline Running...`;
    } else {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="zap"></i> Generate Script & Report`;
    }
    lucide.createIcons();
}

// 4. Stepper & Status Polling
function resetProgressStepper() {
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`step-${i}`);
        step.className = "step";
        step.querySelector(".step-num").innerHTML = `<i data-lucide="circle"></i>`;
    }
    updateConsoleLog("Initiating generation task...", "text-violet");
    lucide.createIcons();
}

function updateConsoleLog(message, colorClass = "") {
    const consoleLogs = document.getElementById("console-logs");
    consoleLogs.className = "console-body " + colorClass;
    consoleLogs.innerHTML = `&gt; ${message}`;
}

function startPollingTaskStatus(taskId) {
    if (pollingInterval) clearInterval(pollingInterval);
    
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/tasks/${taskId}`);
            const task = await response.json();
            
            // Update console text
            updateConsoleLog(task.log, task.status === "failed" ? "text-red" : "text-emerald");
            
            // Update stepper active/completed states
            const currentStage = task.stage;
            for (let i = 1; i <= 6; i++) {
                const step = document.getElementById(`step-${i}`);
                if (i < currentStage) {
                    step.className = "step completed";
                    step.querySelector(".step-num").innerHTML = `<i data-lucide="check"></i>`;
                } else if (i === currentStage) {
                    step.className = "step active";
                    step.querySelector(".step-num").innerHTML = `<i class="spin" data-lucide="refresh-cw"></i>`;
                } else {
                    step.className = "step";
                    step.querySelector(".step-num").innerHTML = `<i data-lucide="circle"></i>`;
                }
            }
            lucide.createIcons();

            if (task.status === "completed") {
                clearInterval(pollingInterval);
                setSubmitButtonState(false);
                document.getElementById("progress-card").classList.add("hidden");
                
                // Fetch and render preview output files
                loadAndRenderPreviews(task.script_file, task.report_file, task.topic);
            } else if (task.status === "failed") {
                clearInterval(pollingInterval);
                setSubmitButtonState(false);
                alert("Generation Pipeline Failed!\nReason: " + task.log);
            }
        } catch (err) {
            console.error("Polling error:", err);
        }
    }, 1200);
}

// 5. Previews Loader & Renderer
async function loadAndRenderPreviews(scriptFilename, reportFilename, topic) {
    try {
        // Fetch Script File
        const scriptRes = await fetch(`/api/history/files/${scriptFilename}`);
        const scriptData = await scriptRes.json();
        currentLoadedScriptText = scriptData.content;
        
        // Fetch Report File
        const reportRes = await fetch(`/api/history/files/${reportFilename}`);
        const reportData = await reportRes.json();
        currentLoadedReportText = reportData.content;

        // Inject content
        document.getElementById("generated-topic-title").textContent = topic;
        
        // Render Markdown content
        const scriptHtml = renderScriptMarkdown(currentLoadedScriptText);
        document.getElementById("script-preview-panel").innerHTML = scriptHtml;
        
        const reportHtml = renderReportMetrics(currentLoadedReportText);
        document.getElementById("report-preview-panel").innerHTML = reportHtml;

        // Reveal Results Area
        document.getElementById("results-preview-section").classList.remove("hidden");
        // Scroll to preview section smoothly
        document.getElementById("results-preview-section").scrollIntoView({ behavior: 'smooth' });
        
        // Reload settings summary key counts
        loadConfigSummary();
    } catch (err) {
        alert("Failed to render preview dashboards: " + err.message);
    }
}

// Custom Markdown Parser for scripts
function renderScriptMarkdown(text) {
    if (!text) return "No script loaded.";
    const lines = text.split("\n");
    let html = [];
    let inBlockquote = false;
    let inList = false;

    for (let line of lines) {
        let trimmed = line.trim();

        // Close list
        if (inList && !trimmed.startsWith("-") && !trimmed.startsWith("*")) {
            html.push("</ul>");
            inList = false;
        }

        // Close blockquote
        if (inBlockquote && !trimmed.startsWith(">")) {
            html.push("</blockquote>");
            inBlockquote = false;
        }

        if (trimmed.startsWith("###")) {
            html.push(`<h3>${trimmed.replace(/^###\s*/, "")}</h3>`);
        } else if (trimmed.startsWith("##")) {
            html.push(`<h2>${trimmed.replace(/^##\s*/, "")}</h2>`);
        } else if (trimmed.startsWith("#")) {
            html.push(`<h1>${trimmed.replace(/^#\s*/, "")}</h1>`);
        } else if (trimmed.startsWith(">")) {
            if (!inBlockquote) {
                html.push("<blockquote>");
                inBlockquote = true;
            }
            html.push(`<p>${trimmed.replace(/^>\s*/, "")}</p>`);
        } else if (trimmed.startsWith("- 🎬") || trimmed.startsWith("- 🎵") || trimmed.startsWith("- ⏱️")) {
            let emoji = trimmed.substring(2, 4);
            let content = trimmed.substring(5);
            let labelClass = trimmed.includes("Visual") ? "text-violet" : trimmed.includes("Audio") ? "text-emerald" : "text-amber";
            html.push(`<p style="margin-left: 1.25rem; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.35rem;"><strong class="${labelClass}">${emoji}</strong> ${content}</p>`);
        } else if (trimmed.startsWith("-") || trimmed.startsWith("*")) {
            if (!inList) {
                html.push("<ul>");
                inList = true;
            }
            html.push(`<li>${trimmed.replace(/^[-*]\s*/, "")}</li>`);
        } else if (trimmed === "") {
            // Spacer
        } else {
            html.push(`<p>${trimmed}</p>`);
        }
    }

    if (inList) html.push("</ul>");
    if (inBlockquote) html.push("</blockquote>");

    return html.join("\n");
}

// Parser for the Growth & Safety report details
function renderReportMetrics(text) {
    if (!text) return "No report loaded.";
    
    // Extract metadata values via Regex
    let monetization = "Safe";
    let copyright = "Safe";
    let score = 85;
    
    const monetMatch = text.match(/Monetization Status:\s*\*\*([A-Za-z]+)\*\*/i);
    if (monetMatch) monetization = monetMatch[1];
    
    const copyMatch = text.match(/Copyright Status:\s*\*\*([A-Za-z]+)\*\*/i);
    if (copyMatch) copyright = copyMatch[1];
    
    const scoreMatch = text.match(/Retention Score:\s*\*\*(\d+)\/100\*\*/i);
    if (scoreMatch) score = parseInt(scoreMatch[1]);

    // Split document segments to extract sections
    const sections = text.split("###");
    let safetyFlagsHtml = "<li>✅ No safety flags raised.</li>";
    let assetsHtml = "";
    let retentionHtml = "";
    let seoTitleHtml = "";
    let tagsHtml = "";
    let chaptersHtml = "";
    let descriptionText = "";

    sections.forEach(sec => {
        const lines = sec.trim().split("\n");
        const header = lines[0].trim().toLowerCase();
        
        if (header.includes("safety & compliance flags")) {
            const listItems = lines.slice(1).map(l => l.trim()).filter(l => l.startsWith("-"));
            if (listItems.length > 0) {
                safetyFlagsHtml = listItems.map(item => `<li>${item.replace(/^- \s*/, "")}</li>`).join("");
            }
        } else if (header.includes("royalty-free asset")) {
            const listItems = lines.slice(1).map(l => l.trim()).filter(l => l.startsWith("-"));
            assetsHtml = listItems.map(item => {
                let cleaned = item.replace(/^-\s*\[\s*\]\s*/, "").replace(/^- \s*/, "");
                return `<li><input type="checkbox" style="margin-right:0.5rem; accent-color:var(--accent-violet);"> ${cleaned}</li>`;
            }).join("");
        } else if (header.includes("engagement & retention")) {
            const listItems = lines.slice(1).map(l => l.trim()).filter(l => l.startsWith("-"));
            retentionHtml = listItems.map(item => `<li>💡 ${item.replace(/^- \s*💡\s*/, "").replace(/^- \s*/, "")}</li>`).join("");
        } else if (header.includes("youtube seo metadata")) {
            // Find titles, tags, chapters, description in sub-lines
            const subtext = lines.slice(1).join("\n");
            
            // Extract titles
            const titleOptions = subtext.match(/\d+\.\s*\*\*(.*?)\*\*/g);
            if (titleOptions) {
                seoTitleHtml = titleOptions.map(opt => `<li>${opt.replace(/^\d+\.\s*\*\*/, "").replace(/\*\*/, "")}</li>`).join("");
            }
            
            // Extract tags
            const tagsMatch = subtext.match(/`([^`]+)`/);
            if (tagsMatch) {
                tagsHtml = tagsMatch[1];
            }
            
            // Extract chapters
            const chapterLines = subtext.split("####");
            chapterLines.forEach(cl => {
                const clLines = cl.trim().split("\n");
                const clHeader = clLines[0].toLowerCase();
                if (clHeader.includes("chapter timestamps")) {
                    const cItems = clLines.slice(1).map(l => l.trim()).filter(l => l.startsWith("-"));
                    chaptersHtml = cItems.map(c => `<li>${c.replace(/^- \s*/, "")}</li>`).join("");
                } else if (clHeader.includes("optimized video description")) {
                    // Extract code block
                    descriptionText = clLines.slice(2, clLines.length - 1).join("\n").trim();
                }
            });
        }
    });

    const monetClass = monetization === "Safe" ? "badge-success" : monetization === "Warning" ? "badge-warning" : "badge-danger";
    const copyClass = copyright === "Safe" ? "badge-success" : "badge-warning";

    return `
        <div class="compliance-score-box">
            <div class="radial-score" style="--score-val: ${score}%">${score}%</div>
            <div class="score-details">
                <h4>Retention & Hook Score</h4>
                <p>Calculated based on early visual resets and hook stakes.</p>
            </div>
        </div>

        <div class="status-badge-row">
            <div class="status-card">
                <span>Monetization Status</span>
                <h4 class="badge ${monetClass}" style="text-align:center; padding: 0.5rem 0.75rem; font-size:0.9rem;">${monetization}</h4>
            </div>
            <div class="status-card">
                <span>Copyright / Fair Use</span>
                <h4 class="badge ${copyClass}" style="text-align:center; padding: 0.5rem 0.75rem; font-size:0.9rem;">${copyright}</h4>
            </div>
        </div>

        <div class="report-section">
            <h4>Safety & Policy Warnings</h4>
            <ul style="color:var(--text-secondary); display:flex; flex-direction:column; gap:0.6rem;">
                ${safetyFlagsHtml}
            </ul>
        </div>

        <div class="report-section">
            <h4>Royalty-Free Asset Checklist</h4>
            <ul>
                ${assetsHtml}
            </ul>
        </div>

        <div class="report-section">
            <h4>Engagement & Watch Time Tips</h4>
            <ul>
                ${retentionHtml}
            </ul>
        </div>

        <div class="report-section" style="border-top: 1px solid rgba(255,255,255,0.05); padding-top:1.5rem;">
            <h4>CTR Title Suggestions</h4>
            <ol style="margin-left: 1.25rem; color:var(--text-secondary); font-size:0.9rem; line-height:1.6;">
                ${seoTitleHtml}
            </ol>
        </div>

        <div class="report-section">
            <h4>Video Search Tags</h4>
            <div class="copy-box">${tagsHtml}</div>
        </div>

        <div class="report-section">
            <h4>Video Chapter Timestamps</h4>
            <ul style="color:var(--text-secondary); font-size:0.9rem; line-height:1.6;">
                ${chaptersHtml}
            </ul>
        </div>

        <div class="report-section">
            <h4>Optimized Video Description</h4>
            <div class="copy-box">${descriptionText}</div>
        </div>
    `;
}

// Copy to Clipboard
function exportToClipboard(type) {
    const text = type === 'script' ? currentLoadedScriptText : currentLoadedReportText;
    if (!text) {
        alert("No content available to copy.");
        return;
    }
    
    navigator.clipboard.writeText(text).then(() => {
        alert(`Copied ${type === 'script' ? 'YouTube script' : 'Growth safety report'} markdown to clipboard!`);
    }).catch(err => {
        console.error("Copy failed:", err);
    });
}

// 6. History Library Controller
// Updated History loader supporting both Single and Series
async function loadHistory() {
    const historyList = document.getElementById("history-list");
    historyList.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);"><i class="spin" data-lucide="refresh-cw" style="margin-right:0.5rem; display:inline-block; vertical-align:middle;"></i> Loading library files...</td></tr>`;
    lucide.createIcons();

    try {
        // Fetch Single Scripts
        const resSingle = await fetch("/api/history");
        const listSingle = await resSingle.json();

        // Fetch Series
        const resSeries = await fetch("/api/series/history");
        const listSeries = await resSeries.json();
        
        // Fetch Planners
        const resPlanner = await fetch("/api/planner/history");
        const listPlanner = await resPlanner.json();
        
        // Merge list
        const mergedList = [];
        listSingle.forEach(item => {
            mergedList.push({
                type: 'single',
                id: item.id,
                title: item.topic,
                niche: item.niche,
                date: item.created_at,
                script_file: item.script_file,
                report_file: item.report_file,
                status_text: "Saved Script"
            });
        });
        
        listSeries.forEach(item => {
            mergedList.push({
                type: 'series',
                id: item.series_id,
                title: `${item.title} (Series)`,
                niche: item.niche,
                date: item.created_at,
                script_file: null,
                report_file: null,
                status_text: `${item.completed_episodes}/${item.total_episodes} Episodes`
            });
        });

        listPlanner.forEach(item => {
            mergedList.push({
                type: 'planner',
                id: item.planner_id,
                title: `Growth Plan: ${item.channel_handle}`,
                niche: item.niche,
                date: item.created_at,
                script_file: null,
                report_file: null,
                status_text: `${item.plan_days}d Calendar`
            });
        });

        // Sort by date descending
        mergedList.sort((a, b) => b.date - a.date);

        if (mergedList.length === 0) {
            historyList.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-secondary); padding: 3rem;">No generated files found in output/ library.</td></tr>`;
            return;
        }

        historyList.innerHTML = mergedList.map(item => {
            const dateStr = new Date(item.date * 1000).toLocaleString();
            
            let previewAction = "";
            let deleteAction = "";
            let badgeClass = "badge-success";
            
            if (item.type === 'series') {
                badgeClass = 'badge-primary';
                previewAction = `previewSeriesItem('${item.id}')`;
                deleteAction = `deleteSeriesItem('${item.id}')`;
            } else if (item.type === 'planner') {
                badgeClass = 'badge-info';
                previewAction = `previewPlannerItem('${item.id}')`;
                deleteAction = `deletePlannerItem('${item.id}')`;
            } else {
                previewAction = `previewHistoryItem('${item.script_file}', '${item.report_file}', '${item.title.replace(/'/g, "\\'")}')`;
                deleteAction = `deleteHistoryItem('${item.id}')`;
            }

            return `
                <tr id="row-${item.id}">
                    <td style="font-weight:600; color:white;">${item.title}</td>
                    <td><span class="badge badge-primary">${item.niche}</span></td>
                    <td style="color:var(--text-secondary); font-size:0.85rem;">${dateStr}</td>
                    <td><span class="badge ${badgeClass}">${item.status_text}</span></td>
                    <td class="actions-cell">
                        <button class="btn-icon" onclick="${previewAction}" title="Preview files">
                            <i data-lucide="eye"></i>
                        </button>
                        <button class="btn-icon btn-icon-danger" onclick="${deleteAction}" title="Delete files">
                            <i data-lucide="trash-2"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join("");
        lucide.createIcons();
    } catch (err) {
        historyList.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-red);">Error loading history: ${err.message}</td></tr>`;
    }
}

function previewHistoryItem(scriptFile, reportFile, topic) {
    // Switch to generator tab to show previews
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    navItems.forEach(nav => {
        if (nav.getAttribute("data-tab") === "generator") nav.classList.add("active");
        else nav.classList.remove("active");
    });

    tabContents.forEach(content => {
        if (content.id === "tab-generator") content.classList.add("active");
        else content.classList.remove("active");
    });

    activeTab = 'generator';
    updateHeaderTitles();

    // Hide processing elements
    document.getElementById("placeholder-card").classList.add("hidden");
    document.getElementById("progress-card").classList.add("hidden");

    loadAndRenderPreviews(scriptFile, reportFile, topic);
}

async function deleteHistoryItem(id) {
    if (!confirm("Are you sure you want to permanently delete these generated script and report files?")) {
        return;
    }

    try {
        const response = await fetch(`/api/history/${id}`, { method: "DELETE" });
        const res = await response.json();
        
        if (res.status === "deleted") {
            const row = document.getElementById(`row-${id}`);
            if (row) row.remove();
            
            // If the deleted script was currently loaded in the generator preview, hide it
            const currentTopic = document.getElementById("generated-topic-title").textContent;
            if (currentTopic.toLowerCase().includes(id.replace(/_/g, " ").toLowerCase())) {
                document.getElementById("results-preview-section").classList.add("hidden");
                document.getElementById("placeholder-card").classList.remove("hidden");
            }
        }
    } catch (err) {
        alert("Failed to delete item: " + err.message);
    }
}

// =====================================================================
// SERIES MODE FRONTEND ENGINE
// =====================================================================

function initSeriesForm() {
    const seriesForm = document.getElementById("series-form");
    if (!seriesForm) return;

    seriesForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const formData = new FormData(seriesForm);
        const numEpisodes = parseInt(formData.get("num_episodes"));

        const payload = {
            title: formData.get("title"),
            series_arc: formData.get("series_arc"),
            niche: formData.get("niche"),
            tone: formData.get("tone"),
            num_episodes: numEpisodes,
            provider: formData.get("provider"),
            model: formData.get("model") || null
        };

        // Reset Stepper
        const stepper = document.getElementById("series-stepper");
        stepper.innerHTML = `
            <div class="step" id="series-step-0">
                <div class="step-num"><i data-lucide="circle"></i></div>
                <div class="step-desc">
                    <h4>Series Bible Design</h4>
                    <p>Creating show outlines, episode topics, and character bibles...</p>
                </div>
            </div>
        `;
        for (let i = 1; i <= numEpisodes; i++) {
            stepper.innerHTML += `
                <div class="step" id="series-step-${i}">
                    <div class="step-num"><i data-lucide="circle"></i></div>
                    <div class="step-desc">
                        <h4>Episode ${i} Script Generation</h4>
                        <p>Drafting spoken narration, cues, safety screenings, and SEO tags...</p>
                    </div>
                </div>
            `;
        }
        lucide.createIcons();

        document.getElementById("series-placeholder-card").classList.add("hidden");
        document.getElementById("series-preview-section").classList.add("hidden");
        document.getElementById("series-progress-card").classList.remove("hidden");

        // Disable button
        const btn = document.getElementById("series-submit-btn");
        btn.disabled = true;
        btn.innerHTML = `<i class="spin" data-lucide="refresh-cw"></i> Generating Series...`;

        try {
            const response = await fetch("/api/series/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            startPollingSeriesStatus(data.task_id, numEpisodes);
        } catch (err) {
            document.getElementById("series-console-logs").className = "console-body text-red";
            document.getElementById("series-console-logs").innerHTML = `&gt; Error: ${err.message}`;
            btn.disabled = false;
            btn.innerHTML = `<i data-lucide="zap"></i> Generate Series Bible & Scripts`;
            lucide.createIcons();
        }
    });
}

function startPollingSeriesStatus(taskId, numEpisodes) {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/series/tasks/${taskId}`);
            const task = await response.json();

            // Log Console
            const consoleLogs = document.getElementById("series-console-logs");
            consoleLogs.className = "console-body " + (task.status === "failed" ? "text-red" : "text-emerald");
            consoleLogs.innerHTML = `&gt; ${task.log}`;

            // Stepper update
            const stage = task.stage; // 0 for bible, 1..N for episodes
            for (let i = 0; i <= numEpisodes; i++) {
                const step = document.getElementById(`series-step-${i}`);
                if (!step) continue;
                
                if (i < stage) {
                    step.className = "step completed";
                    step.querySelector(".step-num").innerHTML = `<i data-lucide="check"></i>`;
                } else if (i === stage) {
                    step.className = "step active";
                    step.querySelector(".step-num").innerHTML = `<i class="spin" data-lucide="refresh-cw"></i>`;
                } else {
                    step.className = "step";
                    step.querySelector(".step-num").innerHTML = `<i data-lucide="circle"></i>`;
                }
            }
            lucide.createIcons();

            if (task.status === "completed") {
                clearInterval(pollingInterval);
                const btn = document.getElementById("series-submit-btn");
                btn.disabled = false;
                btn.innerHTML = `<i data-lucide="zap"></i> Generate Series Bible & Scripts`;
                document.getElementById("series-progress-card").classList.add("hidden");

                loadAndRenderSeriesPreviews(task.series_id);
            } else if (task.status === "failed") {
                clearInterval(pollingInterval);
                const btn = document.getElementById("series-submit-btn");
                btn.disabled = false;
                btn.innerHTML = `<i data-lucide="zap"></i> Generate Series Bible & Scripts`;
                alert("Series Generation Failed!\nReason: " + task.log);
            }
        } catch (err) {
            console.error("Series polling error:", err);
        }
    }, 1200);
}

async function loadAndRenderSeriesPreviews(seriesId) {
    try {
        const response = await fetch(`/api/series/history/${seriesId}`);
        const data = await response.json();
        const bible = data.bible;

        document.getElementById("series-preview-title").textContent = `Series: ${bible.title}`;
        document.getElementById("series-preview-arc").textContent = `Niche: ${bible.niche} | Tone: ${bible.tone} | Learning Arc: ${bible.series_arc}`;

        // Populate Selector Chips
        const chipsContainer = document.getElementById("episode-selector-chips");
        chipsContainer.innerHTML = `<button class="episode-chip active" id="chip-bible" onclick="renderSeriesOutline(${JSON.stringify(bible).replace(/"/g, '&quot;')})"><i data-lucide="book-open" style="width:14px; height:14px; display:inline-block; vertical-align:middle; margin-right:0.25rem;"></i> Series Outline</button>`;
        
        data.episodes.forEach(ep => {
            chipsContainer.innerHTML += `
                <button class="episode-chip" id="chip-ep-${ep.episode_number}" onclick="loadSeriesEpisode('${seriesId}', ${ep.episode_number}, '${ep.script_file}', '${ep.report_file}')">
                    Episode ${ep.episode_number}: ${ep.title}
                </button>
            `;
        });
        lucide.createIcons();

        // Bind active chip states
        const chips = chipsContainer.querySelectorAll(".episode-chip");
        chips.forEach(c => {
            c.addEventListener("click", () => {
                chips.forEach(x => x.classList.remove("active"));
                c.classList.add("active");
            });
        });

        // Set default preview to Series Bible outline
        renderSeriesOutline(bible);

        // Reveal Results Area
        document.getElementById("series-preview-section").classList.remove("hidden");
        document.getElementById("series-preview-section").scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        alert("Failed to render series previews: " + err.message);
    }
}

function renderSeriesOutline(bible) {
    // Left: Series Arc Overview
    const scriptPanel = document.getElementById("series-script-preview");
    scriptPanel.innerHTML = `
        <h1>Series Bible: ${bible.title}</h1>
        <blockquote>
            <p><strong>Niche:</strong> ${bible.niche} | <strong>Tone:</strong> ${bible.tone}</p>
            <p><strong>Series Narrative/Learning Arc:</strong></p>
            <p>${bible.series_arc}</p>
        </blockquote>
        
        <h2>Continuity Notes Log</h2>
        <ul style="margin-left:1.5rem; color:var(--text-secondary); line-height:1.6;">
            ${bible.continuity_notes.map(n => `<li>${n}</li>`).join("")}
        </ul>
    `;

    // Right: Episodes List
    const reportPanel = document.getElementById("series-report-preview");
    reportPanel.innerHTML = `
        <div class="report-section">
            <h4 style="font-size:1.1rem; color:white; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.5rem; margin-bottom:1rem;"><i data-lucide="list" style="width:16px; height:16px; margin-right:0.25rem;"></i> Episode Progression List</h4>
            <div style="display:flex; flex-direction:column; gap:1.25rem;">
                ${bible.episodes.map(ep => `
                    <div style="background-color:rgba(255,255,255,0.02); border:1px solid var(--border-color); padding:1rem; border-radius:8px;">
                        <h4 style="font-size:0.95rem; color:#c084fc; margin-bottom:0.25rem;">Episode ${ep.episode_number}: ${ep.title}</h4>
                        <p style="font-size:0.82rem; color:var(--text-secondary); margin-bottom:0.5rem;"><strong>Focus:</strong> ${ep.focus_topic}</p>
                        <p style="font-size:0.75rem; color:var(--text-muted);"><strong>Takeaways:</strong> ${ep.key_takeaways.join(", ")}</p>
                    </div>
                `).join("")}
            </div>
        </div>
    `;
    lucide.createIcons();
}

async function loadSeriesEpisode(seriesId, epNum, scriptFilename, reportFilename) {
    const scriptPanel = document.getElementById("series-script-preview");
    const reportPanel = document.getElementById("series-report-preview");

    scriptPanel.innerHTML = `<h3><i class="spin" data-lucide="refresh-cw"></i> Loading script content...</h3>`;
    reportPanel.innerHTML = `<h3><i class="spin" data-lucide="refresh-cw"></i> Loading report content...</h3>`;
    lucide.createIcons();

    try {
        // Fetch Script
        const scriptRes = await fetch(`/api/series/history/${seriesId}/files/${scriptFilename}`);
        const scriptData = await scriptRes.json();
        
        // Fetch Report
        const reportRes = await fetch(`/api/series/history/${seriesId}/files/${reportFilename}`);
        const reportData = await reportRes.json();

        // Render Markdown
        scriptPanel.innerHTML = renderScriptMarkdown(scriptData.content);
        reportPanel.innerHTML = renderReportMetrics(reportData.content);
    } catch (err) {
        scriptPanel.innerHTML = `<h3 class="text-red">Error loading files: ${err.message}</h3>`;
        reportPanel.innerHTML = `<h3 class="text-red">Error loading files: ${err.message}</h3>`;
    }
}

// Preview history folder
async function previewSeriesItem(seriesId) {
    // Switch to Series tab
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    navItems.forEach(nav => {
        if (nav.getAttribute("data-tab") === "series") nav.classList.add("active");
        else nav.classList.remove("active");
    });

    tabContents.forEach(content => {
        if (content.id === "tab-series") content.classList.add("active");
        else content.classList.remove("active");
    });

    activeTab = 'series';
    updateHeaderTitles();

    // Hide placeholders
    document.getElementById("series-placeholder-card").classList.add("hidden");
    document.getElementById("series-progress-card").classList.add("hidden");

    loadAndRenderSeriesPreviews(seriesId);
}

// Delete Series folder
async function deleteSeriesItem(seriesId) {
    if (!confirm("Are you sure you want to permanently delete the entire Series folder, including the bible and all episode scripts?")) {
        return;
    }

    try {
        const response = await fetch(`/api/series/history/${seriesId}`, { method: "DELETE" });
        const res = await response.json();

        if (res.status === "deleted") {
            const row = document.getElementById(`row-${seriesId}`);
            if (row) row.remove();

            // If the deleted series is currently open, hide it
            const currentTitle = document.getElementById("series-preview-title").textContent;
            if (currentTitle.toLowerCase().includes(seriesId.replace(/_/g, " ").toLowerCase())) {
                document.getElementById("series-preview-section").classList.add("hidden");
                document.getElementById("series-placeholder-card").classList.remove("hidden");
            }
        }
    } catch (err) {
        alert("Failed to delete series: " + err.message);
    }
}

// =====================================================================
// CHANNEL PLANNER FRONTEND ENGINE
// =====================================================================

function initPlannerForm() {
    const plannerForm = document.getElementById("planner-form");
    if (!plannerForm) return;

    const step1 = document.getElementById("planner-workflow-step-1");
    const step2 = document.getElementById("planner-workflow-step-2");
    const btnNext = document.getElementById("planner-btn-next");
    const btnBack = document.getElementById("planner-btn-back");
    const recommendationsList = document.getElementById("planner-recommendations-list");

    // Next step navigation
    btnNext.addEventListener("click", async () => {
        const channelInput = document.getElementById("planner-channel-url");
        if (channelInput.value && !channelInput.checkValidity()) {
            channelInput.reportValidity();
            return;
        }

        // Switch Step UI
        step1.classList.add("hidden");
        step2.classList.remove("hidden");

        // Load Recommendations
        recommendationsList.innerHTML = `
            <div style="color:var(--text-muted); font-size:0.9rem; text-align:center; padding:1.5rem 0;">
                <i class="spin" data-lucide="refresh-cw" style="display:block; margin:0 auto 0.5rem;"></i> Querying YouTube...
            </div>
        `;
        lucide.createIcons();

        const niche = document.getElementById("planner-niche").value;
        try {
            const response = await fetch(`/api/planner/trending-recommendations?niche=${niche}`);
            const data = await response.json();

            if (data.length === 0) {
                recommendationsList.innerHTML = `<div style="text-align:center; color:var(--text-muted); font-size:0.9rem; padding:1rem 0;">No videos found. Check connection or try another niche.</div>`;
                return;
            }

            recommendationsList.innerHTML = data.map((item, idx) => `
                <label style="display:flex; align-items:flex-start; gap:0.65rem; background:rgba(255,255,255,0.015); border:1px solid var(--border-color); padding:0.65rem 0.85rem; border-radius:6px; cursor:pointer; transition:all 0.2s ease;" class="recommendation-item">
                    <input type="checkbox" class="planner-video-checkbox" value="${item.url}" checked style="margin-top:0.25rem; accent-color:var(--accent-violet);">
                    <div style="flex:1;">
                        <strong style="color:white; font-size:0.82rem; display:block; line-height:1.45;">${item.title}</strong>
                        <span style="font-size:0.75rem; color:var(--text-muted); display:block; margin-top:0.2rem; display:flex; align-items:center; gap:0.25rem;">
                            <i data-lucide="youtube" style="width:12px; height:12px; color:#ef4444;"></i> ID: ${item.video_id}
                        </span>
                    </div>
                </label>
            `).join("");
            lucide.createIcons();
        } catch (err) {
            recommendationsList.innerHTML = `<div style="text-align:center; color:var(--text-red); font-size:0.9rem; padding:1rem 0;">Failed to fetch: ${err.message}</div>`;
        }
    });

    // Back navigation
    btnBack.addEventListener("click", () => {
        step2.classList.add("hidden");
        step1.classList.remove("hidden");
    });

    plannerForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Gather all checked recommendation checkboxes
        const checkboxes = document.querySelectorAll(".planner-video-checkbox:checked");
        const videoUrls = Array.from(checkboxes).map(cb => cb.value);

        if (videoUrls.length === 0) {
            if (!confirm("No competitor videos selected. Proceed with a general niche growth plan?")) {
                return;
            }
        }

        const payload = {
            channel_url: document.getElementById("planner-channel-url").value,
            niche: document.getElementById("planner-niche").value,
            plan_days: parseInt(document.getElementById("planner-days").value),
            video_urls: videoUrls,
            provider: document.getElementById("planner-provider").value,
            model: document.getElementById("planner-model").value || null
        };

        // Reset Stepper
        for (let i = 1; i <= 4; i++) {
            const step = document.getElementById(`planner-step-${i}`);
            if (step) {
                step.className = "step";
                step.querySelector(".step-num").innerHTML = `<i data-lucide="circle"></i>`;
            }
        }
        lucide.createIcons();

        document.getElementById("planner-placeholder-card").classList.add("hidden");
        document.getElementById("planner-result-section").classList.add("hidden");
        document.getElementById("planner-progress-card").classList.remove("hidden");

        const btnSubmit = document.getElementById("planner-submit-btn");
        btnSubmit.disabled = true;
        btnSubmit.innerHTML = `<i class="spin" data-lucide="refresh-cw"></i> Running Pipeline...`;

        try {
            const response = await fetch("/api/planner/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            // Switch view back to Step 1 setup to reset state
            step2.classList.add("hidden");
            step1.classList.remove("hidden");
            
            startPollingPlannerStatus(data.task_id);
        } catch (err) {
            document.getElementById("planner-console-logs").className = "console-body text-red";
            document.getElementById("planner-console-logs").innerHTML = `&gt; Error: ${err.message}`;
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = `<i data-lucide="compass"></i> Audit & Plan`;
            lucide.createIcons();
        }
    });
}

function startPollingPlannerStatus(taskId) {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/planner/tasks/${taskId}`);
            const task = await response.json();

            // Log Console
            const consoleLogs = document.getElementById("planner-console-logs");
            consoleLogs.className = "console-body " + (task.status === "failed" ? "text-red" : "text-emerald");
            consoleLogs.innerHTML = `&gt; ${task.log}`;

            // Stepper update
            const stage = task.stage; // 1 to 4
            for (let i = 1; i <= 4; i++) {
                const step = document.getElementById(`planner-step-${i}`);
                if (!step) continue;
                
                if (i < stage) {
                    step.className = "step completed";
                    step.querySelector(".step-num").innerHTML = `<i data-lucide="check"></i>`;
                } else if (i === stage) {
                    step.className = "step active";
                    step.querySelector(".step-num").innerHTML = `<i class="spin" data-lucide="refresh-cw"></i>`;
                } else {
                    step.className = "step";
                    step.querySelector(".step-num").innerHTML = `<i data-lucide="circle"></i>`;
                }
            }
            lucide.createIcons();

            if (task.status === "completed") {
                clearInterval(pollingInterval);
                const btn = document.getElementById("planner-submit-btn");
                btn.disabled = false;
                btn.innerHTML = `<i data-lucide="compass"></i> Analyze & Generate Growth Plan`;
                document.getElementById("planner-progress-card").classList.add("hidden");

                loadAndRenderPlannerPreviews(task.planner_id);
            } else if (task.status === "failed") {
                clearInterval(pollingInterval);
                const btn = document.getElementById("planner-submit-btn");
                btn.disabled = false;
                btn.innerHTML = `<i data-lucide="compass"></i> Analyze & Generate Growth Plan`;
                alert("Channel Analysis Failed!\nReason: " + task.log);
            }
        } catch (err) {
            console.error("Planner polling error:", err);
        }
    }, 1200);
}

async function loadAndRenderPlannerPreviews(plannerId) {
    try {
        const response = await fetch(`/api/planner/history/${plannerId}`);
        const data = await response.json();

        document.getElementById("planner-result-title").textContent = `Growth Plan for ${data.channel_handle}`;
        document.getElementById("planner-result-meta").textContent = `Niche: ${data.niche} | Horizon: ${data.plan_days} Days`;

        // Render Gaps
        const gapsList = document.getElementById("planner-gaps-list");
        gapsList.innerHTML = data.niche_gaps.map(g => `<li>${g}</li>`).join("");

        // Render Opportunities
        const oppList = document.getElementById("planner-opportunities-list");
        oppList.innerHTML = data.opportunities.map(o => `<li>${o}</li>`).join("");

        // Render Audits
        const auditsContainer = document.getElementById("planner-audits-container");
        const auditsCard = document.getElementById("planner-audits-card");
        if (data.audited_videos && data.audited_videos.length > 0) {
            auditsCard.classList.remove("hidden");
            auditsContainer.innerHTML = data.audited_videos.map(audit => `
                <div class="planner-audit-item">
                    <h5 style="color:white; font-size:0.95rem; margin-bottom:0.25rem;"><i data-lucide="youtube" style="width:14px; height:14px; color:#ef4444; margin-right:0.25rem; display:inline-block; vertical-align:middle;"></i> ${audit.title}</h5>
                    <p style="font-size:0.8rem; color:var(--text-secondary);"><strong>Video ID:</strong> <a href="https://youtube.com/watch?v=${audit.video_id}" target="_blank" style="color:#60a5fa;">${audit.video_id}</a></p>
                    <p style="font-size:0.82rem; color:var(--text-secondary); margin-top:0.25rem;"><strong>Content Delivered:</strong> ${audit.content_delivered}</p>
                    <p style="font-size:0.82rem; color:var(--text-secondary);"><strong>Delivery Breakdown:</strong> ${audit.delivery_breakdown}</p>
                    <p style="font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;"><strong>Hooks Identified:</strong> ${audit.hooks_used.join(" | ")}</p>
                </div>
            `).join("");
        } else {
            auditsCard.classList.add("hidden");
        }

        // Render Calendar list
        const calendarContainer = document.getElementById("planner-calendar-container");
        calendarContainer.innerHTML = data.content_calendar.map(item => {
            const formatBadge = item.format.includes("Short") 
                ? `<span class="badge badge-primary" style="background-color:#db2777; border-color:#db2777;">${item.format}</span>`
                : `<span class="badge badge-primary">${item.format}</span>`;
                
            return `
                <div class="planner-calendar-day">
                    <div style="flex:1;">
                        <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
                            <span class="badge" style="background-color:rgba(255,255,255,0.06); color:white; border:1px solid rgba(255,255,255,0.1);">Day ${item.day}</span>
                            ${formatBadge}
                            <strong style="color:white; font-size:0.95rem;">${item.title}</strong>
                        </div>
                        <p style="font-size:0.85rem; color:var(--text-secondary); margin-top:0.35rem;"><strong>Suggested Angle:</strong> ${item.angle}</p>
                        <p style="font-size:0.8rem; color:var(--text-muted); margin-top:0.15rem;"><strong>Niche Gap Rationale:</strong> ${item.niche_gap_reason}</p>
                    </div>
                    <button class="btn btn-secondary btn-sm" onclick="createScriptFromPlan('${item.title.replace(/'/g, "\\'")}', '${data.niche}')" style="white-space:nowrap;">
                        <i data-lucide="sparkles" style="width:12px; height:12px; margin-right:0.25rem;"></i> Create Script
                    </button>
                </div>
            `;
        }).join("");
        lucide.createIcons();

        // Reveal result section
        document.getElementById("planner-result-section").classList.remove("hidden");
        document.getElementById("planner-result-section").scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        alert("Failed to render planner: " + err.message);
    }
}

// Redirect planned video title to script generator form
function createScriptFromPlan(title, niche) {
    // 1. Populate Generator fields
    document.getElementById("topic").value = title;
    
    const nicheSelect = document.getElementById("niche");
    if (nicheSelect) {
        nicheSelect.value = niche;
        populateNicheSuggestions(); // Update suggestion chips for the niche!
    }

    // 2. Switch to generator tab
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    navItems.forEach(nav => {
        if (nav.getAttribute("data-tab") === "generator") nav.classList.add("active");
        else nav.classList.remove("active");
    });

    tabContents.forEach(content => {
        if (content.id === "tab-generator") content.classList.add("active");
        else content.classList.remove("active");
    });

    activeTab = 'generator';
    updateHeaderTitles();

    // Scroll to form topic field
    document.getElementById("topic").focus();
    document.getElementById("topic").scrollIntoView({ behavior: 'smooth' });
}

// Preview history planner item
async function previewPlannerItem(plannerId) {
    // Switch to Planner tab
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    navItems.forEach(nav => {
        if (nav.getAttribute("data-tab") === "planner") nav.classList.add("active");
        else nav.classList.remove("active");
    });

    tabContents.forEach(content => {
        if (content.id === "tab-planner") content.classList.add("active");
        else content.classList.remove("active");
    });

    activeTab = 'planner';
    updateHeaderTitles();

    // Hide placeholders
    document.getElementById("planner-placeholder-card").classList.add("hidden");
    document.getElementById("planner-progress-card").classList.add("hidden");

    loadAndRenderPlannerPreviews(plannerId);
}

// Delete history planner item
async function deletePlannerItem(plannerId) {
    if (!confirm("Are you sure you want to permanently delete this Channel Plan audit report?")) {
        return;
    }

    try {
        const response = await fetch(`/api/planner/history/${plannerId}`, { method: "DELETE" });
        const res = await response.json();

        if (res.status === "deleted") {
            const row = document.getElementById(`row-${plannerId}`);
            if (row) row.remove();

            // If deleted planner was open, hide it
            const currentTitle = document.getElementById("planner-result-title").textContent;
            if (currentTitle.toLowerCase().includes(plannerId.split("_")[0].toLowerCase())) {
                document.getElementById("planner-result-section").classList.add("hidden");
                document.getElementById("planner-placeholder-card").classList.remove("hidden");
            }
        }
    } catch (err) {
        alert("Failed to delete planner: " + err.message);
    }
}
