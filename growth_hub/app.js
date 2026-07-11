/**
 * Zenith Growth Hub - Core Application Logic
 *persists state, renders simulations, runs sandboxed test cases, and controls navigation.
 */

// ==========================================
// 1. Initial State & State Management
// ==========================================
window.appState = {
  xp: 0,
  level: 1,
  streak: 0,
  lastActiveDate: null,
  goals: [
    { id: 1, text: "Solve 1 Competitive Programming problem", completed: false },
    { id: 2, text: "Practice speaking about Singleton Pattern using CEB", completed: false },
    { id: 3, text: "Explore Eigenvectors visualizer in Linear Algebra", completed: false }
  ],
  completedProblems: [],
  speakingSessionsCount: 0,
  
  // Save to LocalStorage
  save() {
    localStorage.setItem('zenith_growth_hub_state', JSON.stringify({
      xp: this.xp,
      level: this.level,
      streak: this.streak,
      lastActiveDate: this.lastActiveDate,
      goals: this.goals,
      completedProblems: this.completedProblems,
      speakingSessionsCount: this.speakingSessionsCount
    }));
    this.updateUI();
  },

  // Load from LocalStorage
  load() {
    const saved = localStorage.getItem('zenith_growth_hub_state');
    if (saved) {
      const data = JSON.parse(saved);
      this.xp = data.xp || 0;
      this.level = data.level || 1;
      this.streak = data.streak || 0;
      this.lastActiveDate = data.lastActiveDate || null;
      this.goals = data.goals || this.goals;
      this.completedProblems = data.completedProblems || [];
      this.speakingSessionsCount = data.speakingSessionsCount || 0;
    }
    this.checkStreak();
    this.updateUI();
  },

  // Streak checking logic
  checkStreak() {
    const today = new Date().toDateString();
    if (!this.lastActiveDate) {
      this.streak = 1;
      this.lastActiveDate = today;
    } else {
      const lastDate = new Date(this.lastActiveDate);
      const todayDate = new Date(today);
      const diffTime = Math.abs(todayDate - lastDate);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      
      if (diffDays === 1) {
        this.streak += 1;
        this.lastActiveDate = today;
      } else if (diffDays > 1) {
        this.streak = 1; // Streak reset
        this.lastActiveDate = today;
      }
    }
  },

  addXP(amount) {
    this.xp += amount;
    const xpNeeded = this.level * 100;
    if (this.xp >= xpNeeded) {
      this.xp -= xpNeeded;
      this.level += 1;
      showNotification(`🎉 LEVEL UP! You reached Level ${this.level}!`, 'success');
    }
    this.save();
  },

  // Sync state to UI elements
  updateUI() {
    const xpNeeded = this.level * 100;
    const xpPct = (this.xp / xpNeeded) * 100;
    
    // Sidebar
    document.getElementById('user-level-label').innerText = `Level ${this.level}`;
    document.getElementById('user-xp-bar').style.width = `${xpPct}%`;
    document.getElementById('user-xp-text').innerText = `${this.xp} / ${xpNeeded} XP`;
    document.getElementById('streak-days').innerText = this.streak;
    
    // Dashboard Skills percentages
    const totalMath = 4; // calculus, linear algebra, probability, math-in-ai
    const totalCP = window.cpProblems ? window.cpProblems.length : 5;
    const cpDone = this.completedProblems.length;
    
    // Rough progress representations
    const englishProgress = Math.min(100, this.speakingSessionsCount * 20);
    const mathProgress = 50; // Visual baseline, since it's sandbox
    const cpProgress = totalCP ? Math.round((cpDone / totalCP) * 100) : 0;
    const designProgress = 40; // Static placeholder indicating read documentation
    
    document.getElementById('stat-english-progress').innerText = `${englishProgress}%`;
    document.getElementById('bar-english-progress').style.width = `${englishProgress}%`;
    
    document.getElementById('stat-math-progress').innerText = `${mathProgress}%`;
    document.getElementById('bar-math-progress').style.width = `${mathProgress}%`;
    
    document.getElementById('stat-cp-progress').innerText = `${cpProgress}%`;
    document.getElementById('bar-cp-progress').style.width = `${cpProgress}%`;
    
    document.getElementById('stat-design-progress').innerText = `${designProgress}%`;
    document.getElementById('bar-design-progress').style.width = `${designProgress}%`;
    
    // Rerender goals checklist
    this.renderGoals();
  },

  renderGoals() {
    const container = document.getElementById('goals-list-container');
    if (!container) return;
    container.innerHTML = '';
    
    this.goals.forEach(goal => {
      const li = document.createElement('li');
      li.className = 'goal-item';
      li.innerHTML = `
        <div class="goal-item-left">
          <input type="checkbox" id="goal-${goal.id}" ${goal.completed ? 'checked' : ''}>
          <span class="goal-text ${goal.completed ? 'completed' : ''}">${goal.text}</span>
        </div>
        <button class="btn-delete-goal" onclick="window.appState.deleteGoal(${goal.id})"><i class="fa-solid fa-trash"></i></button>
      `;
      // Toggle logic
      li.querySelector('input').addEventListener('change', (e) => {
        goal.completed = e.target.checked;
        if (goal.completed) {
          this.addXP(15);
          showNotification("+15 XP Earned!", "success");
        }
        this.save();
      });
      container.appendChild(li);
    });
  },

  addGoal(text) {
    if (!text.trim()) return;
    const newId = this.goals.length ? Math.max(...this.goals.map(g => g.id)) + 1 : 1;
    this.goals.push({ id: newId, text: text, completed: false });
    this.save();
  },

  deleteGoal(id) {
    this.goals = this.goals.filter(g => g.id !== id);
    this.save();
  },

  // Randomized motivational quotes
  quotes: [
    `"The only way to learn mathematics is to do mathematics." — Paul Halmos`,
    `"Linear algebra is the mathematics of the 21st century." — Gilbert Strang`,
    `"The quietest people have the loudest minds." — Stephen Hawking`,
    `"Simplicity is prerequisite for reliability." — Edsger W. Dijkstra`,
    `"First, solve the problem. Then, write the code." — John Johnson`,
    `"If you want to find the secrets of the universe, think in terms of energy, frequency and vibration." — Nikola Tesla`,
    `"Focus on signal over noise. Don't waste time on things that don't actually make things better." — Elon Musk`
  ],

  getNewQuote() {
    const rand = Math.floor(Math.random() * this.quotes.length);
    document.getElementById('motivational-quote').innerText = this.quotes[rand];
  }
};

// Simple global notification helper
function showNotification(message, type = 'success') {
  const container = document.body;
  const toast = document.createElement('div');
  toast.style.position = 'fixed';
  toast.style.bottom = '20px';
  toast.style.right = '20px';
  toast.style.padding = '1rem 1.5rem';
  toast.style.background = type === 'success' ? '#10b981' : '#ef4444';
  toast.style.color = '#000';
  toast.style.fontWeight = 'bold';
  toast.style.borderRadius = '8px';
  toast.style.zIndex = '9999';
  toast.style.boxShadow = '0 4px 12px rgba(0,0,0,0.5)';
  toast.style.animation = 'slideDown 0.3s ease';
  toast.innerText = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}


// ==========================================
// 2. Global Navigation Handler
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
  // Navigation Tabs
  const navItems = document.querySelectorAll('.nav-item');
  const tabContents = document.querySelectorAll('.tab-content');
  
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetTab = item.getAttribute('data-tab');
      
      navItems.forEach(n => n.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      item.classList.add('active');
      const targetElement = document.getElementById(targetTab);
      if (targetElement) {
        targetElement.classList.add('active');
      }
      
      // Trigger canvas drawing updates if switching to their tabs
      if (targetTab === 'math') {
        setTimeout(() => {
          updateRiemannCanvas();
          updateEigenCanvas();
          updateProbCanvas();
        }, 100);
      } else if (targetTab === 'system-design') {
        setTimeout(() => {
          initInfraSimulator();
        }, 100);
      }
    });
  });

  // Goal adding in UI
  const addGoalBtn = document.getElementById('add-goal-btn');
  const goalInput = document.getElementById('new-goal-input');
  if (addGoalBtn && goalInput) {
    addGoalBtn.addEventListener('click', () => {
      window.appState.addGoal(goalInput.value);
      goalInput.value = '';
    });
    goalInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        window.appState.addGoal(goalInput.value);
        goalInput.value = '';
      }
    });
  }

  // Load state
  window.appState.load();
});


// ==========================================
// 3. Technical Speaking Gym Logic
// ==========================================
window.speakingPrompts = {
  architecture: [
    {
      text: "Explain how a CDN (Content Delivery Network) speeds up asset loading.",
      keywords: ["edge server", "caching", "latency", "geographical distance", "origin server", "bandwidth"]
    },
    {
      text: "Explain the difference between vertical scaling and horizontal scaling.",
      keywords: ["scale out", "single point of failure", "load balancer", "elasticity", "hardware limits", "stateless server"]
    },
    {
      text: "What happens under the hood when a client makes a read request served by Redis?",
      keywords: ["in-memory database", "cache hit", "cache miss", "read-through caching", "round-trip time", "disk storage fallback"]
    }
  ],
  coding: [
    {
      text: "Explain how a Hash Map resolves collisions under the hood.",
      keywords: ["hash function", "buckets", "chaining", "linked list", "red-black tree", "load factor", "O(1) average case"]
    },
    {
      text: "Explain the differences between DFS and BFS and when you would use each.",
      keywords: ["stack", "queue", "shortest path", "depth-first search", "breadth-first search", "backtracking", "space complexity"]
    },
    {
      text: "Explain the Sieve of Eratosthenes algorithm for finding prime numbers.",
      keywords: ["prime numbers", "composite numbers", "array of booleans", "marking multiples", "O(N log log N)"]
    }
  ],
  design: [
    {
      text: "Explain the Singleton design pattern. When would you use it, and what are its drawbacks?",
      keywords: ["single instance", "global access point", "private constructor", "lazy initialization", "testing difficulties", "concurrency synchronization"]
    },
    {
      text: "Explain the Observer design pattern and how it decouples objects.",
      keywords: ["subject", "subscribers", "loose coupling", "event listener", "state change notification", "one-to-many relationship"]
    },
    {
      text: "What is the Dependency Inversion Principle (the D in SOLID) and why is it useful?",
      keywords: ["high-level module", "low-level module", "interfaces / abstractions", "decoupling", "mocking for tests", "injection framework"]
    }
  ],
  behavioral: [
    {
      text: "Describe a time you encountered a difficult technical challenge and how you solved it.",
      keywords: ["analytical diagnosis", "root cause", "alternative trade-offs", "collaboration", "solution implementation", "post-mortem takeaway"]
    },
    {
      text: "How do you explain a highly complex technical concept to a non-technical stakeholder?",
      keywords: ["analogy", "business value outcome", "avoiding jargon", "active listening", "feedback checkpoints", "visual abstractions"]
    }
  ],
  explain: [
    {
      text: "Explain how a Database Index works to a non-technical product manager.",
      keywords: ["book index", "alphabetical search", "scanning pages", "lookup speed", "avoiding full table scan", "write overhead"],
      analogy: "Think of a database index like the index at the back of a textbook. Instead of flipping through every single page to find 'Recursion' (a full table scan), you look it up in the index, find the exact page number, and jump directly there. It makes reads super fast, but writing a new page is slightly slower because you have to update the index too."
    },
    {
      text: "Explain how public-key cryptography (HTTPS/SSL) works to a high schooler.",
      keywords: ["padlock", "public key", "private key", "secure box", "encryption", "web browser"],
      analogy: "Imagine I want you to send me a secret letter. I send you an open padlock but keep the key. You write the letter, put it in a box, snap my padlock shut, and send it back. No one who intercepts the box can open it, not even you, because only I have the key. In HTTPS, your browser uses the website's public padlock to lock data, and only the website can unlock it."
    },
    {
      text: "Explain what a memory leak is to a business analyst.",
      keywords: ["unused resources", "clogged memory", "system slow down", "restaurant tables", "releasing reference", "rebooting"],
      analogy: "Imagine running a busy restaurant. Customers sit down, eat, and leave, but the waiters forget to clear the tables (memory leaks). As the day goes on, new customers arrive but find no clean tables, so they wait in long lines, and the restaurant slows to a crawl. Eventually, the restaurant has to close and reboot (restart the server) to clear all tables."
    },
    {
      text: "Explain why refactoring technical debt is necessary to a non-technical project manager.",
      keywords: ["interest rate", "clean kitchen", "future features", "speed of delivery", "code rot", "long-term stability"],
      analogy: "Think of coding like cooking in a kitchen. Technical debt is like cooking a meal and leaving the dirty pans in the sink. If you keep cooking meals (adding features) without washing the dishes (refactoring), eventually the kitchen becomes so messy and crowded that it takes twice as long to make even a simple sandwich. Refactoring is washing the dishes so we can cook faster next week."
    },
    {
      text: "Explain the difference between SQL and NoSQL databases to a new intern.",
      keywords: ["relational", "structured table", "schema flexibility", "excel sheet", "json documents", "scaling out"],
      analogy: "SQL is like a pre-formatted Excel spreadsheet: every row must have the exact same columns (Name, Age, Email). It's clean and structured. NoSQL is like a folder of loose paper documents (JSON files): one page might have Name and Age, another might have Name, Address, and Hobbies. SQL is strict and relational, while NoSQL is flexible and scales out easily."
    }
  ]
};

let activePromptCategory = 'architecture';
let activePromptIndex = 0;
let activeFramework = 'star';
let timerInterval = null;
let timerSeconds = 0;
let evaluatedScore = { clarity: 0, structure: 0, pacing: 0, confidence: 0 };

document.addEventListener('DOMContentLoaded', () => {
  // Category switching in Speaking Gym
  const categoryBtns = document.querySelectorAll('.category-btn');
  categoryBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      categoryBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activePromptCategory = btn.getAttribute('data-cat');
      loadSpeakingPrompts();
    });
  });

  // Framework card switching
  const frameworkCards = document.querySelectorAll('.framework-card');
  frameworkCards.forEach(card => {
    card.addEventListener('click', () => {
      frameworkCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      activeFramework = card.getAttribute('data-framework');
      updateFrameworkHint();
    });
  });

  // Timer controls
  const startBtn = document.getElementById('timer-start-btn');
  const pauseBtn = document.getElementById('timer-pause-btn');
  const resetBtn = document.getElementById('timer-reset-btn');
  
  if (startBtn) {
    startBtn.addEventListener('click', startTimer);
    pauseBtn.addEventListener('click', pauseTimer);
    resetBtn.addEventListener('click', finishTimerPractice);
  }

  // Star scoring logic
  const starContainers = document.querySelectorAll('.stars');
  starContainers.forEach(container => {
    const stars = container.querySelectorAll('i');
    const metric = container.getAttribute('data-metric');
    
    stars.forEach(star => {
      star.addEventListener('click', () => {
        const val = parseInt(star.getAttribute('data-value'));
        evaluatedScore[metric] = val;
        
        // Highlight active stars
        stars.forEach(s => {
          const sVal = parseInt(s.getAttribute('data-value'));
          if (sVal <= val) {
            s.className = 'fa-solid fa-star';
          } else {
            s.className = 'fa-regular fa-star';
          }
        });
        
        // Enable log button if all metrics are rated
        checkScoreCompletion();
      });
    });
  });

  // Log Score Button
  const submitScoreBtn = document.getElementById('submit-score-btn');
  if (submitScoreBtn) {
    submitScoreBtn.addEventListener('click', () => {
      const avgRating = (evaluatedScore.clarity + evaluatedScore.structure + evaluatedScore.pacing + evaluatedScore.confidence) / 4;
      const baseXP = 30;
      const bonusXP = Math.round(avgRating * 10);
      const earnedXP = baseXP + bonusXP;
      
      window.appState.speakingSessionsCount += 1;
      window.appState.addXP(earnedXP);
      
      showNotification(`Logged session! Earned ${earnedXP} XP!`, 'success');
      
      // Reset gym state
      document.getElementById('evaluation-card').classList.add('hidden');
      resetTimer();
      submitScoreBtn.disabled = true;
      
      // Clear stars
      document.querySelectorAll('.stars i').forEach(s => s.className = 'fa-regular fa-star');
      Object.keys(evaluatedScore).forEach(k => evaluatedScore[k] = 0);
    });
  }

  // Load first set of prompts
  loadSpeakingPrompts();
});

function loadSpeakingPrompts() {
  const container = document.getElementById('speaking-prompts-container');
  if (!container) return;
  container.innerHTML = '';
  
  const prompts = window.speakingPrompts[activePromptCategory] || [];
  prompts.forEach((p, idx) => {
    const card = document.createElement('div');
    card.className = `prompt-item-card ${idx === activePromptIndex ? 'active' : ''}`;
    card.innerText = p.text;
    card.addEventListener('click', () => {
      document.querySelectorAll('.prompt-item-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      activePromptIndex = idx;
      updateActivePromptDisplay();
    });
    container.appendChild(card);
  });
  
  // Auto update active display
  updateActivePromptDisplay();
}

function updateActivePromptDisplay() {
  const promptData = window.speakingPrompts[activePromptCategory][activePromptIndex];
  if (!promptData) return;
  
  document.getElementById('active-prompt-category').innerText = activePromptCategory.toUpperCase();
  document.getElementById('active-prompt-text').innerText = promptData.text;
  
  // Keyword Tags loading
  const keywordContainer = document.getElementById('keyword-pool-container');
  if (keywordContainer) {
    keywordContainer.innerHTML = '';
    promptData.keywords.forEach(kw => {
      const tag = document.createElement('div');
      tag.className = 'keyword-tag';
      tag.innerHTML = `<i class="fa-regular fa-circle"></i> <span>${kw}</span>`;
      tag.addEventListener('click', () => {
        tag.classList.toggle('checked');
        const icon = tag.querySelector('i');
        if (tag.classList.contains('checked')) {
          icon.className = 'fa-solid fa-circle-check';
        } else {
          icon.className = 'fa-regular fa-circle';
        }
      });
      keywordContainer.appendChild(tag);
    });
  }
  
  updateFrameworkHint();
}

function updateFrameworkHint() {
  const box = document.getElementById('framework-hint');
  if (!box) return;
  
  const promptData = window.speakingPrompts[activePromptCategory][activePromptIndex];
  
  let html = '';
  if (activeFramework === 'star') {
    html = `
      <strong>Framework Guide (STAR):</strong>
      <ul>
        <li><strong>Situation</strong>: Set the scene (e.g., users far from server face high latency).</li>
        <li><strong>Task</strong>: What you need to solve (reducing latency for global asset delivery).</li>
        <li><strong>Action</strong>: Cache assets on Edge servers geographically closer to users.</li>
        <li><strong>Result</strong>: Assets load instantly, reducing server load and improving UX.</li>
      </ul>
    `;
  } else if (activeFramework === 'ceb') {
    html = `
      <strong>Framework Guide (Concept-Example-Benefit):</strong>
      <ul>
        <li><strong>Concept</strong>: Define the pattern or technology abstractly (e.g. Singleton maintains a single instance).</li>
        <li><strong>Example</strong>: Provide a real-world coding case (e.g., a database connection pool or system settings manager).</li>
        <li><strong>Benefit</strong>: Detail the value it adds (conserves socket connections, guarantees consistency).</li>
      </ul>
    `;
  } else if (activeFramework === 'eli5') {
    html = `
      <strong>Framework Guide (ELI5 - Explain Like I'm 5):</strong>
      <ul>
        <li><strong>Core Analogy</strong>: Replace all engineering terms (API, memory, database, thread) with everyday physical objects (mailmen, restaurant tables, books, pipes).</li>
        <li><strong>Simplification</strong>: Strip out secondary details. Focus on the 'Why' and the 'How' in one simple sentence.</li>
      </ul>
    `;
    if (promptData && promptData.analogy) {
      html += `
        <div style="margin-top: 10px; padding: 10px; background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.15); border-radius: 8px;">
          <strong style="color: var(--accent-cyan); display: flex; align-items: center; gap: 5px;"><i class="fa-solid fa-lightbulb"></i> Recommended Analogy:</strong>
          <p style="font-size: 0.85rem; line-height: 1.45; color: var(--text-main); margin-top: 5px;">${promptData.analogy}</p>
        </div>
      `;
    }
  } else if (activeFramework === 'prep') {
    html = `
      <strong>Framework Guide (PREP Method):</strong>
      <ul>
        <li><strong>Point</strong>: State your main message clearly and immediately (e.g., "We should use Redis cache because...").</li>
        <li><strong>Reason</strong>: Back up your point (e.g., "It reduces query latency from 150ms to 5ms by keeping data in memory").</li>
        <li><strong>Example</strong>: Provide a real-world scenario (e.g., "For instance, our profile page loads 20 times faster for users").</li>
        <li><strong>Point</strong>: Restate your main message to summarize (e.g., "So, adopting Redis will directly solve our performance bottlenecks").</li>
      </ul>
    `;
  }
  
  box.innerHTML = html;
}

function startTimer() {
  document.getElementById('timer-start-btn').classList.add('hidden');
  document.getElementById('timer-pause-btn').classList.remove('hidden');
  document.getElementById('timer-reset-btn').disabled = false;
  
  timerInterval = setInterval(() => {
    timerSeconds++;
    updateTimerDisplay();
  }, 1000);
}

function pauseTimer() {
  document.getElementById('timer-pause-btn').classList.add('hidden');
  document.getElementById('timer-start-btn').classList.remove('hidden');
  clearInterval(timerInterval);
}

function resetTimer() {
  clearInterval(timerInterval);
  timerSeconds = 0;
  updateTimerDisplay();
  
  document.getElementById('timer-start-btn').classList.remove('hidden');
  document.getElementById('timer-pause-btn').classList.add('hidden');
  document.getElementById('timer-reset-btn').disabled = true;
  
  // Uncheck keywords
  document.querySelectorAll('.keyword-tag').forEach(tag => {
    tag.classList.remove('checked');
    tag.querySelector('i').className = 'fa-regular fa-circle';
  });
}

function finishTimerPractice() {
  pauseTimer();
  // Show evaluation card
  document.getElementById('evaluation-card').classList.remove('hidden');
}

function updateTimerDisplay() {
  const m = Math.floor(timerSeconds / 60).toString().padStart(2, '0');
  const s = (timerSeconds % 60).toString().padStart(2, '0');
  document.getElementById('speaking-timer').innerText = `${m}:${s}`;
}

function checkScoreCompletion() {
  const ready = Object.values(evaluatedScore).every(val => val > 0);
  const btn = document.getElementById('submit-score-btn');
  if (btn) {
    btn.disabled = !ready;
  }
}


// ==========================================
// 4. Mathematics Playground Visualizers
// ==========================================

// --- Math Tab Switcher ---
document.addEventListener('DOMContentLoaded', () => {
  const subTabBtns = document.querySelectorAll('.math-tab-btn');
  const subContents = document.querySelectorAll('.math-subcontent');
  
  subTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      subTabBtns.forEach(b => b.classList.remove('active'));
      subContents.forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      const sub = btn.getAttribute('data-subtab');
      document.getElementById(sub).classList.add('active');
      
      // Update canvas scales
      setTimeout(() => {
        if (sub === 'calculus') updateRiemannCanvas();
        else if (sub === 'linear-algebra') updateEigenCanvas();
        else if (sub === 'probability') updateProbCanvas();
      }, 50);
    });
  });

  // Hook Riemann slider
  const rSlider = document.getElementById('riemann-n-slider');
  if (rSlider) {
    rSlider.addEventListener('input', (e) => {
      document.getElementById('riemann-n-val').innerText = e.target.value;
      updateRiemannCanvas();
    });
  }

  // Hook Eigenvector angle slider
  const eSlider = document.getElementById('eigen-angle-slider');
  if (eSlider) {
    eSlider.addEventListener('input', (e) => {
      document.getElementById('eigen-angle-val').innerText = `${e.target.value}°`;
      updateEigenCanvas();
    });
  }

  // Hook Bayes sliders
  const bPrior = document.getElementById('bayes-prior');
  const bSens = document.getElementById('bayes-sens');
  const bFpr = document.getElementById('bayes-fpr');
  
  if (bPrior) {
    const updateBayes = () => {
      document.getElementById('bayes-prior-val').innerText = `${bPrior.value}%`;
      document.getElementById('bayes-sens-val').innerText = `${bSens.value}%`;
      document.getElementById('bayes-fpr-val').innerText = `${bFpr.value}%`;
      calculateBayes();
    };
    bPrior.addEventListener('input', updateBayes);
    bSens.addEventListener('input', updateBayes);
    bFpr.addEventListener('input', updateBayes);
    calculateBayes();
  }
  
  // Probability distributions UI hooks
  const distBtns = document.querySelectorAll('.dist-btn');
  distBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      distBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setupProbSliders(btn.getAttribute('data-dist'));
    });
  });
  setupProbSliders('normal'); // Default init
});

// --- Calculus: Riemann Sum Visualizer ---
function updateRiemannCanvas() {
  const canvas = document.getElementById('riemann-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  
  ctx.clearRect(0, 0, W, H);
  
  // Math bounds
  const xMin = -0.5, xMax = 2.5;
  const yMin = -0.5, yMax = 4.5;
  
  // Mapping functions
  const toScreenX = (x) => ((x - xMin) / (xMax - xMin)) * W;
  const toScreenY = (y) => H - ((y - yMin) / (yMax - yMin)) * H;
  
  // Draw axes
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(toScreenX(xMin), toScreenY(0));
  ctx.lineTo(toScreenX(xMax), toScreenY(0));
  ctx.moveTo(toScreenX(0), toScreenY(yMin));
  ctx.lineTo(toScreenX(0), toScreenY(yMax));
  ctx.stroke();
  
  // Draw labels
  ctx.fillStyle = '#6b7280';
  ctx.font = '10px monospace';
  ctx.fillText('0', toScreenX(0) - 10, toScreenY(0) + 12);
  ctx.fillText('1', toScreenX(1) - 3, toScreenY(0) + 12);
  ctx.fillText('2', toScreenX(2) - 3, toScreenY(0) + 12);
  ctx.fillText('4', toScreenX(0) - 14, toScreenY(4) + 4);
  
  // Function: f(x) = x^2
  const f = (x) => x * x;
  
  // Partition calculations
  const a = 0, b = 2; // Domain of integration
  const n = parseInt(document.getElementById('riemann-n-slider').value);
  const dx = (b - a) / n;
  
  let approxArea = 0;
  
  // Draw rectangles
  ctx.fillStyle = 'rgba(168, 85, 247, 0.25)';
  ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
  ctx.lineWidth = 1;
  
  for (let i = 0; i < n; i++) {
    // Right hand Riemann Sum
    const rectX1 = a + i * dx;
    const rectX2 = rectX1 + dx;
    const rectY = f(rectX2); // Right endpoint height
    
    approxArea += rectY * dx;
    
    // Canvas dimensions
    const sx1 = toScreenX(rectX1);
    const sx2 = toScreenX(rectX2);
    const sy = toScreenY(rectY);
    const sZero = toScreenY(0);
    
    ctx.fillRect(sx1, sy, sx2 - sx1, sZero - sy);
    ctx.strokeRect(sx1, sy, sx2 - sx1, sZero - sy);
  }
  
  // Plot true function curve
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 3;
  ctx.shadowColor = '#00e5ff';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  
  const step = 0.02;
  let first = true;
  for (let x = a - 0.2; x <= b + 0.2; x += step) {
    const sx = toScreenX(x);
    const sy = toScreenY(f(x));
    if (first) {
      ctx.moveTo(sx, sy);
      first = false;
    } else {
      ctx.lineTo(sx, sy);
    }
  }
  ctx.stroke();
  ctx.shadowBlur = 0; // Reset glow
  
  // Exact Area under x^2 from 0 to 2 is 8/3 = 2.6666...
  const exactArea = 8 / 3;
  const errorPct = Math.abs((approxArea - exactArea) / exactArea) * 100;
  
  // Update texts
  document.getElementById('riemann-approx').innerText = approxArea.toFixed(4);
  document.getElementById('riemann-exact').innerText = exactArea.toFixed(4);
  document.getElementById('riemann-error').innerText = `${errorPct.toFixed(2)}%`;
}

// --- Linear Algebra: Eigenvector Alignment ---
function updateEigenCanvas() {
  const canvas = document.getElementById('eigen-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  
  ctx.clearRect(0, 0, W, H);
  
  // Grid properties
  const size = Math.min(W, H) - 40;
  const cx = W / 2;
  const cy = H / 2;
  const scale = size / 8; // Grid spans -4 to +4
  
  // Draw grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;
  for (let g = -4; g <= 4; g++) {
    ctx.beginPath();
    ctx.moveTo(cx + g * scale, 0); ctx.lineTo(cx + g * scale, H);
    ctx.moveTo(0, cy + g * scale); ctx.lineTo(W, cy + g * scale);
    ctx.stroke();
  }
  
  // Draw major Axes
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
  ctx.moveTo(0, cy); ctx.lineTo(W, cy);
  ctx.stroke();
  
  // User Input Vector v: angle determined by slider
  const angleDeg = parseInt(document.getElementById('eigen-angle-slider').value);
  const angleRad = (angleDeg * Math.PI) / 180;
  
  const vx = Math.cos(angleRad);
  const vy = Math.sin(angleRad);
  
  // Transformation Matrix A = [[2, 1], [1, 2]]
  // w = A * v
  const wx = 2 * vx + 1 * vy;
  const wy = 1 * vx + 2 * vy;
  
  // Draw standard Eigenvector lines for guidelines
  // Eigenvector 1: angle = 45 degrees, lambda = 3
  // Eigenvector 2: angle = 135 degrees, lambda = 1
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(cx - 3.5 * scale, cy + 3.5 * scale); ctx.lineTo(cx + 3.5 * scale, cy - 3.5 * scale);
  ctx.moveTo(cx - 3.5 * scale, cy - 3.5 * scale); ctx.lineTo(cx + 3.5 * scale, cy + 3.5 * scale);
  ctx.stroke();
  ctx.setLineDash([]); // Reset
  
  // Draw transformed vector w (Purple)
  ctx.strokeStyle = '#a855f7';
  ctx.lineWidth = 4;
  ctx.shadowColor = '#a855f7';
  ctx.shadowBlur = 6;
  drawArrow(ctx, cx, cy, cx + wx * scale, cy - wy * scale, 12);
  
  // Draw input vector v (Cyan)
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 3;
  ctx.shadowColor = '#00e5ff';
  ctx.shadowBlur = 6;
  drawArrow(ctx, cx, cy, cx + vx * scale, cy - vy * scale, 8);
  ctx.shadowBlur = 0; // Reset
  
  // Text labels
  ctx.fillStyle = '#00e5ff';
  ctx.font = '11px sans-serif';
  ctx.fillText('v (Input)', cx + vx * scale + 10, cy - vy * scale - 5);
  ctx.fillStyle = '#a855f7';
  ctx.fillText('Av (Transformed)', cx + wx * scale + 10, cy - wy * scale - 5);
  
  // Alignment Check: cross product of v and w
  // If v and w are collinear, cross product = vx*wy - vy*wx = 0
  const cross = Math.abs(vx * wy - vy * wx);
  const statusBox = document.getElementById('eigen-status-box');
  
  if (cross < 0.02) {
    statusBox.className = 'eigen-status aligned';
    // Calculate lambda
    const lambda = Math.sqrt(wx*wx + wy*wy) / Math.sqrt(vx*vx + vy*vy);
    statusBox.innerText = `🌟 EIGENVECTOR ALIGNED! Eigenvalue (λ) = ${lambda.toFixed(1)} (Angle: ${angleDeg}°)`;
    
    // Add particle feedback aura
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, 2*Math.PI);
    ctx.stroke();
  } else {
    statusBox.className = 'eigen-status';
    statusBox.innerText = `Rotate vector v to align its direction with transformed vector Av (Cross Prod error: ${cross.toFixed(3)})`;
  }
}

// Arrow drawer helper
function drawArrow(ctx, fromx, fromy, tox, toy, r) {
  ctx.beginPath();
  ctx.moveTo(fromx, fromy);
  ctx.lineTo(tox, toy);
  ctx.stroke();
  
  const angle = Math.atan2(toy - fromy, tox - fromx);
  ctx.beginPath();
  ctx.moveTo(tox, toy);
  ctx.lineTo(tox - r * Math.cos(angle - Math.PI / 6), toy - r * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(tox - r * Math.cos(angle + Math.PI / 6), toy - r * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
}

// Matrix operations step-by-step calculator
window.appState.calculateMatrix = function() {
  const a = parseFloat(document.getElementById('m-00').value);
  const b = parseFloat(document.getElementById('m-01').value);
  const c = parseFloat(document.getElementById('m-10').value);
  const d = parseFloat(document.getElementById('m-11').value);
  
  const resultDiv = document.getElementById('matrix-results');
  if (!resultDiv) return;
  
  const det = a * d - b * c;
  
  let luHTML = '';
  if (a === 0) {
    luHTML = `<span class="error">LU factorisation requires pivoting since A[0,0] = 0.</span>`;
  } else {
    const l21 = c / a;
    const u22 = d - l21 * b;
    
    luHTML = `
      <div style="margin-top: 8px;">
        <strong>LU Decomposition:</strong><br>
        L = \\begin{pmatrix} 1 & 0 \\\\ ${l21.toFixed(2)} & 1 \\end{pmatrix}<br>
        U = \\begin{pmatrix} ${a.toFixed(2)} & ${b.toFixed(2)} \\\\ 0 & ${u22.toFixed(2)} \\end{pmatrix}
      </div>
    `;
  }
  
  resultDiv.innerHTML = `
    <div><strong>Determinant:</strong> ${det.toFixed(2)}</div>
    ${luHTML}
  `;
  
  // Re-run KaTeX formatting inside matrix results
  renderMathInElement(resultDiv);
};

// --- Probability Distribution Plotter ---
let activeDist = 'normal';
let distParams = {
  normal: { mean: 0, std: 1 },
  uniform: { min: -2, max: 2 },
  exponential: { lambda: 1 },
  poisson: { lambda: 4 },
  binomial: { n: 10, p: 0.5 }
};

function setupProbSliders(dist) {
  activeDist = dist;
  const container = document.getElementById('dist-sliders-container');
  if (!container) return;
  
  container.innerHTML = '';
  
  const params = distParams[dist];
  
  if (dist === 'normal') {
    container.innerHTML = `
      <div class="slider-row">
        <label>Mean (μ): <strong id="param-mu">${params.mean}</strong></label>
        <input type="range" id="slider-mu" min="-3" max="3" step="0.5" value="${params.mean}">
      </div>
      <div class="slider-row">
        <label>Std Dev (σ): <strong id="param-sigma">${params.std}</strong></label>
        <input type="range" id="slider-sigma" min="0.2" max="2" step="0.1" value="${params.std}">
      </div>
    `;
    const sMu = document.getElementById('slider-mu');
    const sSig = document.getElementById('slider-sigma');
    const upd = () => {
      params.mean = parseFloat(sMu.value);
      params.std = parseFloat(sSig.value);
      document.getElementById('param-mu').innerText = params.mean;
      document.getElementById('param-sigma').innerText = params.std;
      updateProbCanvas();
    };
    sMu.addEventListener('input', upd);
    sSig.addEventListener('input', upd);
    
  } else if (dist === 'uniform') {
    container.innerHTML = `
      <div class="slider-row">
        <label>Lower Bound (a): <strong id="param-u-a">${params.min}</strong></label>
        <input type="range" id="slider-u-a" min="-4" max="0" step="0.5" value="${params.min}">
      </div>
      <div class="slider-row">
        <label>Upper Bound (b): <strong id="param-u-b">${params.max}</strong></label>
        <input type="range" id="slider-u-b" min="0.5" max="4" step="0.5" value="${params.max}">
      </div>
    `;
    const sA = document.getElementById('slider-u-a');
    const sB = document.getElementById('slider-u-b');
    const upd = () => {
      params.min = parseFloat(sA.value);
      params.max = parseFloat(sB.value);
      document.getElementById('param-u-a').innerText = params.min;
      document.getElementById('param-u-b').innerText = params.max;
      updateProbCanvas();
    };
    sA.addEventListener('input', upd);
    sB.addEventListener('input', upd);
    
  } else if (dist === 'exponential') {
    container.innerHTML = `
      <div class="slider-row">
        <label>Rate parameter (λ): <strong id="param-exp-lam">${params.lambda}</strong></label>
        <input type="range" id="slider-exp-lam" min="0.2" max="3" step="0.1" value="${params.lambda}">
      </div>
    `;
    const sL = document.getElementById('slider-exp-lam');
    const upd = () => {
      params.lambda = parseFloat(sL.value);
      document.getElementById('param-exp-lam').innerText = params.lambda;
      updateProbCanvas();
    };
    sL.addEventListener('input', upd);
    
  } else if (dist === 'poisson') {
    container.innerHTML = `
      <div class="slider-row">
        <label>Average rate (λ): <strong id="param-pois-lam">${params.lambda}</strong></label>
        <input type="range" id="slider-pois-lam" min="1" max="15" step="1" value="${params.lambda}">
      </div>
    `;
    const sL = document.getElementById('slider-pois-lam');
    const upd = () => {
      params.lambda = parseInt(sL.value);
      document.getElementById('param-pois-lam').innerText = params.lambda;
      updateProbCanvas();
    };
    sL.addEventListener('input', upd);
    
  } else if (dist === 'binomial') {
    container.innerHTML = `
      <div class="slider-row">
        <label>Number of trials (n): <strong id="param-bin-n">${params.n}</strong></label>
        <input type="range" id="slider-bin-n" min="2" max="25" step="1" value="${params.n}">
      </div>
      <div class="slider-row">
        <label>Success probability (p): <strong id="param-bin-p">${params.p}</strong></label>
        <input type="range" id="slider-bin-p" min="0.1" max="0.9" step="0.05" value="${params.p}">
      </div>
    `;
    const sN = document.getElementById('slider-bin-n');
    const sP = document.getElementById('slider-bin-p');
    const upd = () => {
      params.n = parseInt(sN.value);
      params.p = parseFloat(sP.value);
      document.getElementById('param-bin-n').innerText = params.n;
      document.getElementById('param-bin-p').innerText = params.p;
      updateProbCanvas();
    };
    sN.addEventListener('input', upd);
    sP.addEventListener('input', upd);
  }
  
  updateProbCanvas();
}

function updateProbCanvas() {
  const canvas = document.getElementById('prob-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  
  ctx.clearRect(0, 0, W, H);
  
  // Layout bounds mapping
  const xMin = -5, xMax = 5;
  const yMin = -0.05, yMax = 1.05;
  
  const toScreenX = (x) => ((x - xMin) / (xMax - xMin)) * W;
  const toScreenY = (y) => H - ((y - yMin) / (yMax - yMin)) * H;
  
  // Axes drawing
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toScreenX(xMin), toScreenY(0)); ctx.lineTo(toScreenX(xMax), toScreenY(0));
  ctx.moveTo(toScreenX(0), toScreenY(yMin)); ctx.lineTo(toScreenX(0), toScreenY(yMax));
  ctx.stroke();
  
  ctx.fillStyle = '#6b7280';
  ctx.font = '10px monospace';
  ctx.fillText('0', toScreenX(0) - 10, toScreenY(0) + 12);
  ctx.fillText('-4', toScreenX(-4) - 5, toScreenY(0) + 12);
  ctx.fillText('4', toScreenX(4) - 3, toScreenY(0) + 12);
  
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#a855f7';
  ctx.shadowColor = '#a855f7';
  
  let meanVal = 0;
  
  if (activeDist === 'normal') {
    const mu = distParams.normal.mean;
    const s = distParams.normal.std;
    meanVal = mu;
    
    // Normal PDF: 1 / (s * sqrt(2pi)) * e^-(x-mu)^2/2s^2
    const pdf = (x) => (1 / (s * Math.sqrt(2 * Math.PI))) * Math.exp(-Math.pow(x - mu, 2) / (2 * s * s));
    
    ctx.shadowBlur = 6;
    ctx.beginPath();
    let first = true;
    for (let x = xMin; x <= xMax; x += 0.05) {
      const sx = toScreenX(x);
      const sy = toScreenY(pdf(x));
      if (first) { ctx.moveTo(sx, sy); first = false; }
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    
  } else if (activeDist === 'uniform') {
    const a = distParams.uniform.min;
    const b = distParams.uniform.max;
    meanVal = (a + b) / 2;
    
    const height = 1 / (b - a);
    
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.moveTo(toScreenX(xMin), toScreenY(0));
    ctx.lineTo(toScreenX(a), toScreenY(0));
    ctx.lineTo(toScreenX(a), toScreenY(height));
    ctx.lineTo(toScreenX(b), toScreenY(height));
    ctx.lineTo(toScreenX(b), toScreenY(0));
    ctx.lineTo(toScreenX(xMax), toScreenY(0));
    ctx.stroke();
    
  } else if (activeDist === 'exponential') {
    const lam = distParams.exponential.lambda;
    meanVal = 1 / lam;
    
    const pdf = (x) => x < 0 ? 0 : lam * Math.exp(-lam * x);
    
    ctx.shadowBlur = 6;
    ctx.beginPath();
    let first = true;
    for (let x = -1; x <= xMax; x += 0.05) {
      const sx = toScreenX(x);
      const sy = toScreenY(pdf(x));
      if (first) { ctx.moveTo(sx, sy); first = false; }
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    
  } else if (activeDist === 'poisson') {
    const lam = distParams.poisson.lambda;
    meanVal = lam;
    
    // Factorial helper
    const fact = (n) => n <= 1 ? 1 : n * fact(n - 1);
    const pmf = (k) => (Math.pow(lam, k) * Math.exp(-lam)) / fact(k);
    
    // Draw discrete bars from 0 to 15
    ctx.shadowBlur = 4;
    ctx.fillStyle = '#a855f7';
    for (let k = 0; k <= 15; k++) {
      const p = pmf(k);
      const sx = toScreenX(k / 1.5); // Scaled to fit
      const sy = toScreenY(p);
      const sZero = toScreenY(0);
      ctx.fillRect(sx - 5, sy, 10, sZero - sy);
    }
    
  } else if (activeDist === 'binomial') {
    const n = distParams.binomial.n;
    const p = distParams.binomial.p;
    meanVal = n * p;
    
    // Binomial coefficient
    const comb = (n, k) => {
      const f = (num) => num <= 1 ? 1 : num * f(num - 1);
      return f(n) / (f(k) * f(n - k));
    };
    const pmf = (k) => comb(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
    
    ctx.shadowBlur = 4;
    ctx.fillStyle = '#a855f7';
    for (let k = 0; k <= n; k++) {
      const prob = pmf(k);
      const sx = toScreenX((k / n) * 8 - 4); // Stretch -4 to 4
      const sy = toScreenY(prob);
      const sZero = toScreenY(0);
      ctx.fillRect(sx - 5, sy, 10, sZero - sy);
    }
  }
  
  ctx.shadowBlur = 0; // Reset
  
  // Draw Mean Indicator Line (dotted cyan)
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  let screenMeanX;
  if (activeDist === 'binomial') {
    screenMeanX = toScreenX((meanVal / distParams.binomial.n) * 8 - 4);
  } else if (activeDist === 'poisson') {
    screenMeanX = toScreenX(meanVal / 1.5);
  } else {
    screenMeanX = toScreenX(meanVal);
  }
  
  ctx.moveTo(screenMeanX, toScreenY(0));
  ctx.lineTo(screenMeanX, toScreenY(1));
  ctx.stroke();
  ctx.setLineDash([]);
  
  ctx.fillStyle = '#00e5ff';
  ctx.fillText(`μ (Mean): ${meanVal.toFixed(2)}`, screenMeanX + 5, 20);
}

// Bayes Theorem Calculations
function calculateBayes() {
  const prior = parseFloat(document.getElementById('bayes-prior').value) / 100;
  const sens = parseFloat(document.getElementById('bayes-sens').value) / 100;
  const fpr = parseFloat(document.getElementById('bayes-fpr').value) / 100;
  
  const priorHealthy = 1 - prior;
  const totalPositive = (sens * prior) + (fpr * priorHealthy);
  const posterior = (sens * prior) / totalPositive;
  
  const resultsBox = document.getElementById('bayes-results');
  if (resultsBox) {
    resultsBox.innerHTML = `
      <p>Prior (Disease occurs in general): <strong>P(D) = ${(prior*100).toFixed(1)}%</strong></p>
      <p>Likelihood (Test positive if sick): <strong>P(T+|D) = ${(sens*100).toFixed(0)}%</strong></p>
      <p>False Pos (Test positive if healthy): <strong>P(T+|H) = ${(fpr*100).toFixed(0)}%</strong></p>
      <hr style="border: 0; border-top: 1px solid var(--card-border); margin: 8px 0;">
      <p>Overall positive test rate: <strong>P(T+) = ${(totalPositive*100).toFixed(2)}%</strong></p>
      <p style="font-size: 1.05rem; margin-top: 8px;">Probability you are actually sick given positive test:<br>
      <strong style="color: var(--accent-cyan); font-size: 1.25rem;">P(D | T+) = ${(posterior*100).toFixed(2)}%</strong></p>
      <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">Notice how even with 90% test sensitivity, a low general disease rate (1%) makes a positive test yield only a ~15% chance of actual sickness due to False Positives.</p>
    `;
  }
}


// ==========================================
// 5. Competitive Programming Arena Engine
// ==========================================
window.cpProblems = [
  {
    id: 1,
    title: "Two Sum",
    difficulty: "easy",
    category: "array",
    fnName: "twoSum",
    description: `Given an array of integers <code>nums</code> and an integer <code>target</code>, return indices of the two numbers such that they add up to <code>target</code>.<br><br>
    You may assume that each input would have exactly one solution, and you may not use the same element twice.`,
    starter: `function twoSum(nums, target) {
  // Write your code here
  
}`,
    testCases: [
      { input: [[2, 7, 11, 15], 9], expected: [0, 1] },
      { input: [[3, 2, 4], 6], expected: [1, 2] },
      { input: [[3, 3], 6], expected: [0, 1] }
    ]
  },
  {
    id: 2,
    title: "Fibonacci Numbers",
    difficulty: "easy",
    category: "math",
    fnName: "fib",
    description: `The Fibonacci numbers, commonly denoted <code>F(n)</code> form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1.<br><br>
    Given <code>n</code>, calculate <code>F(n)</code>.`,
    starter: `function fib(n) {
  // Write your code here
  
}`,
    testCases: [
      { input: [2], expected: 1 },
      { input: [3], expected: 2 },
      { input: [4], expected: 3 },
      { input: [9], expected: 34 }
    ]
  },
  {
    id: 3,
    title: "Sieve of Eratosthenes Count",
    difficulty: "easy",
    category: "math",
    fnName: "countPrimes",
    description: `Given an integer <code>n</code>, return the number of prime numbers that are strictly less than <code>n</code>.<br><br>
    For example, <code>n = 10</code> should return 4 because there are four prime numbers less than 10 (2, 3, 5, 7).`,
    starter: `function countPrimes(n) {
  // Write your code here
  
}`,
    testCases: [
      { input: [10], expected: 4 },
      { input: [0], expected: 0 },
      { input: [1], expected: 0 },
      { input: [2], expected: 0 }
    ]
  },
  {
    id: 4,
    title: "Binary Search Index",
    difficulty: "easy",
    category: "array",
    fnName: "search",
    description: `Given an array of integers <code>nums</code> which is sorted in ascending order, and an integer <code>target</code>, write a function to search <code>target</code> in <code>nums</code>.<br><br>
    If <code>target</code> exists, then return its index. Otherwise, return -1.`,
    starter: `function search(nums, target) {
  // Write your code here
  
}`,
    testCases: [
      { input: [[-1, 0, 3, 5, 9, 12], 9], expected: 4 },
      { input: [[-1, 0, 3, 5, 9, 12], 2], expected: -1 }
    ]
  },
  {
    id: 5,
    title: "Contains Duplicate",
    difficulty: "easy",
    category: "array",
    fnName: "containsDuplicate",
    description: `Given an integer array <code>nums</code>, return <code>true</code> if any value appears at least twice in the array, and return <code>false</code> if every element is distinct.`,
    starter: `function containsDuplicate(nums) {
  // Write your code here
  
}`,
    testCases: [
      { input: [[1, 2, 3, 1]], expected: true },
      { input: [[1, 2, 3, 4]], expected: false },
      { input: [[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]], expected: true }
    ]
  },
  {
    id: 6,
    title: "Valid Anagram",
    difficulty: "easy",
    category: "string",
    fnName: "isAnagram",
    description: `Given two strings <code>s</code> and <code>t</code>, return <code>true</code> if <code>t</code> is an anagram of <code>s</code>, and <code>false</code> otherwise.<br><br>
    An anagram is a word formed by rearranging the letters of another (e.g., "anagram" and "nagaram").`,
    starter: `function isAnagram(s, t) {
  // Write your code here
  
}`,
    testCases: [
      { input: ["anagram", "nagaram"], expected: true },
      { input: ["rat", "car"], expected: false }
    ]
  },
  {
    id: 7,
    title: "Greatest Common Divisor (GCD)",
    difficulty: "easy",
    category: "math",
    fnName: "gcd",
    description: `Given two positive integers <code>a</code> and <code>b</code>, return their Greatest Common Divisor (GCD) using the Euclidean algorithm.`,
    starter: `function gcd(a, b) {
  // Write your code here
  
}`,
    testCases: [
      { input: [8, 12], expected: 4 },
      { input: [17, 13], expected: 1 },
      { input: [100, 10], expected: 10 }
    ]
  },
  {
    id: 8,
    title: "Fizz Buzz Array",
    difficulty: "easy",
    category: "math",
    fnName: "fizzBuzz",
    description: `Given an integer <code>n</code>, return an array of strings representing answer: <br>
    - <code>answer[i] === "FizzBuzz"</code> if divisible by 3 and 5.<br>
    - <code>answer[i] === "Fizz"</code> if divisible by 3.<br>
    - <code>answer[i] === "Buzz"</code> if divisible by 5.<br>
    - <code>answer[i] === i.toString()</code> (1-indexed) otherwise.`,
    starter: `function fizzBuzz(n) {
  // Write your code here
  
}`,
    testCases: [
      { input: [3], expected: ["1", "2", "Fizz"] },
      { input: [5], expected: ["1", "2", "Fizz", "4", "Buzz"] },
      { input: [15], expected: ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"] }
    ]
  },
  {
    id: 9,
    title: "Maximum Subarray Sum",
    difficulty: "medium",
    category: "array",
    fnName: "maxSubArray",
    description: `Given an integer array <code>nums</code>, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum (Kadane's algorithm).`,
    starter: `function maxSubArray(nums) {
  // Write your code here
  
}`,
    testCases: [
      { input: [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], expected: 6 },
      { input: [[1]], expected: 1 },
      { input: [[5, 4, -1, 7, 8]], expected: 23 }
    ]
  }
];

let activeProblemId = 1;

document.addEventListener('DOMContentLoaded', () => {
  loadProblemsList();
  
  // Hook workspace buttons
  const runBtn = document.getElementById('cp-run-btn');
  const submitBtn = document.getElementById('cp-submit-btn');
  
  if (runBtn) {
    runBtn.addEventListener('click', runCPTestCases);
    submitBtn.addEventListener('click', submitCPSolution);
  }
});

function loadProblemsList() {
  const container = document.getElementById('cp-problem-list');
  if (!container) return;
  container.innerHTML = '';
  
  window.cpProblems.forEach(p => {
    const isCompleted = window.appState.completedProblems.includes(p.id);
    const item = document.createElement('div');
    item.className = `problem-list-item ${p.id === activeProblemId ? 'active' : ''} ${isCompleted ? 'completed' : 'uncompleted'}`;
    item.innerHTML = `
      <div>
        <div class="title">${p.title}</div>
        <div class="difficulty">${p.difficulty}</div>
      </div>
      <i class="fa-solid ${isCompleted ? 'fa-circle-check' : 'fa-circle'}"></i>
    `;
    item.addEventListener('click', () => {
      document.querySelectorAll('.problem-list-item').forEach(el => el.classList.remove('active'));
      item.classList.add('active');
      activeProblemId = p.id;
      loadProblemToWorkspace();
    });
    container.appendChild(item);
  });
  
  loadProblemToWorkspace();
}

function loadProblemToWorkspace() {
  const p = window.cpProblems.find(prob => prob.id === activeProblemId);
  if (!p) return;
  
  document.getElementById('cp-prob-title').innerText = p.title;
  document.getElementById('cp-prob-desc').innerHTML = p.description;
  
  const savedCode = localStorage.getItem(`cp_code_${p.id}`);
  document.getElementById('code-textarea').value = savedCode || p.starter;
  
  // Clear terminal
  document.getElementById('cp-terminal').innerText = "Console output will appear here after running tests.";
  document.getElementById('cp-submit-btn').disabled = true;
}

window.appState.resetCode = function() {
  const p = window.cpProblems.find(prob => prob.id === activeProblemId);
  if (!p) return;
  document.getElementById('code-textarea').value = p.starter;
  localStorage.removeItem(`cp_code_${p.id}`);
};

// Safe sandbox evaluator
function runCPTestCases() {
  const code = document.getElementById('code-textarea').value;
  const p = window.cpProblems.find(prob => prob.id === activeProblemId);
  if (!p) return;
  
  // Save progress code
  localStorage.setItem(`cp_code_${p.id}`, code);
  
  const term = document.getElementById('cp-terminal');
  term.innerHTML = "Running tests...\n\n";
  
  let allPassed = true;
  
  // Re-eval context sandbox wrapper
  try {
    // Basic parser checks to avoid infinite loops in raw submissions
    if (code.includes('while(true)') || code.includes('while (true)')) {
      throw new Error("Potential infinite loop detected (while true). Execution blocked.");
    }
    
    // Construct execution context
    let functionName = p.fnName;
    
    // Inject user code and retrieve reference
    const creator = new Function(code + `\nreturn ${functionName};`);
    const fnRef = creator();
    
    if (typeof fnRef !== 'function') {
      throw new Error(`Function ${functionName} is not defined. Make sure you don't rename the starter function.`);
    }
    
    // Run test cases
    p.testCases.forEach((tc, index) => {
      const inputs = JSON.parse(JSON.stringify(tc.input)); // Deep copy
      const result = fnRef(...inputs);
      const expected = tc.expected;
      
      const isMatch = Array.isArray(expected) 
        ? Array.isArray(result) && result.length === expected.length && result.every((v, i) => v === expected[i])
        : result === expected;
        
      if (isMatch) {
        term.innerHTML += `<span class="success">✓ Test Case ${index + 1} Passed!</span>\n`;
        term.innerHTML += `  Input: ${JSON.stringify(tc.input)}\n`;
        term.innerHTML += `  Output: ${JSON.stringify(result)}\n\n`;
      } else {
        term.innerHTML += `<span class="error">✗ Test Case ${index + 1} Failed!</span>\n`;
        term.innerHTML += `  Input: ${JSON.stringify(tc.input)}\n`;
        term.innerHTML += `  Expected: ${JSON.stringify(expected)}\n`;
        term.innerHTML += `  Got: ${JSON.stringify(result)}\n\n`;
        allPassed = false;
      }
    });
    
  } catch (err) {
    term.innerHTML += `<span class="error">Syntax or Runtime Error: ${err.message}</span>\n`;
    allPassed = false;
  }
  
  if (allPassed) {
    term.innerHTML += `<span class="success" style="font-weight:bold;">All test cases passed! Ready to submit.</span>`;
    document.getElementById('cp-submit-btn').disabled = false;
  } else {
    document.getElementById('cp-submit-btn').disabled = true;
  }
}

function submitCPSolution() {
  const p = window.cpProblems.find(prob => prob.id === activeProblemId);
  if (!p) return;
  
  if (!window.appState.completedProblems.includes(p.id)) {
    window.appState.completedProblems.push(p.id);
    window.appState.addXP(40); // Earn 40 XP
    showNotification("Solution Submitted! +40 XP Earned!", "success");
    loadProblemsList(); // Rerender
  } else {
    showNotification("Already submitted previously, no new XP awarded.");
  }
}


// ==========================================
// 6. System Design & Design Patterns Vault
// ==========================================

// --- SOLID principles database ---
window.solidPrinciples = {
  s: {
    title: "S: Single Responsibility Principle",
    desc: "A class should have one, and only one, reason to change. It means a component should do exactly one thing.",
    dirty: `// DIRTY: User class handles data, database writes, and email notifications
class User {
  constructor(name, email) {
    this.name = name;
    this.email = email;
  }
  
  saveToDatabase() {
    console.log("Saving user to DB...");
  }
  
  sendWelcomeEmail() {
    console.log("Sending email to " + this.email);
  }
}`,
    clean: `// CLEAN: Responsibilities decoupled into helper services
class User {
  constructor(name, email) {
    this.name = name;
    this.email = email;
  }
}

class UserRepository {
  save(user) {
    console.log("Saving user to DB...");
  }
}

class EmailService {
  sendWelcome(user) {
    console.log("Sending email to " + user.email);
  }
}`
  },
  o: {
    title: "O: Open/Closed Principle",
    desc: "Software entities (classes, modules, functions) should be open for extension, but closed for modification.",
    dirty: `// DIRTY: Adding a new shape requires modifying the Calculator class
class AreaCalculator {
  calculate(shapes) {
    return shapes.reduce((area, shape) => {
      if (shape.type === 'circle') {
        return area + Math.PI * shape.radius ** 2;
      } else if (shape.type === 'rectangle') {
        return area + shape.width * shape.height;
      }
    }, 0);
  }
}`,
    clean: `// CLEAN: Shapes implement their own Area getter, closed to changes
class Circle {
  constructor(radius) { this.radius = radius; }
  area() { return Math.PI * this.radius ** 2; }
}

class Rectangle {
  constructor(w, h) { this.width = w; this.height = h; }
  area() { return this.width * this.height; }
}

class AreaCalculator {
  calculate(shapes) {
    return shapes.reduce((sum, shape) => sum + shape.area(), 0);
  }
}`
  },
  l: {
    title: "L: Liskov Substitution Principle",
    desc: "Subtypes must be substitutable for their base types without altering correctness of the program.",
    dirty: `// DIRTY: Ostrich inherits Bird but breaks fly functionality
class Bird {
  fly() { console.log("Flying..."); }
}

class Ostrich extends Bird {
  fly() {
    throw new Error("Ostriches cannot fly!"); // Violates LSP!
  }
}`,
    clean: `// CLEAN: Break properties into separate interfaces/subclasses
class Bird {}

class FlyingBird extends Bird {
  fly() { console.log("Flying..."); }
}

class Ostrich extends Bird {
  run() { console.log("Running..."); }
}`
  },
  i: {
    title: "I: Interface Segregation Principle",
    desc: "Clients should not be forced to depend on methods they do not use. Prefer many small client-specific interfaces.",
    dirty: `// DIRTY: Normal worker forced to implement robot-specific recharge methods
class WorkerInterface {
  work() {}
  eat() {}
  recharge() {} // Humans do not recharge batteries!
}`,
    clean: `// CLEAN: Segregated properties into small interfaces
class Workable {
  work() {}
}

class Feedable {
  eat() {}
}

class Rechargeable {
  recharge() {}
}`
  },
  d: {
    title: "D: Dependency Inversion Principle",
    desc: "High-level modules should not depend on low-level modules. Both should depend on abstractions.",
    dirty: `// DIRTY: Car high-level module depends directly on V8Engine concrete details
class V8Engine {
  start() { console.log("Vroom..."); }
}

class Car {
  constructor() {
    this.engine = new V8Engine(); // Tight coupling!
  }
  drive() { this.engine.start(); }
}`,
    clean: `// CLEAN: Depend on Interface abstraction, injected at constructor
class Car {
  constructor(engine) { // Injected dependency
    this.engine = engine; 
  }
  drive() { this.engine.start(); }
}`
  }
};

// --- Design Patterns Database ---
window.designPatterns = {
  singleton: {
    title: "Singleton Pattern",
    desc: "Ensures that a class has only one instance and provides a global point of access to it. Highly used in database connection pools or system configuration objects.",
    usecase: `// Practical Javascript Singleton implementation
class DatabaseConnection {
  constructor() {
    if (DatabaseConnection.instance) {
      return DatabaseConnection.instance;
    }
    this.connectionString = "postgresql://localhost:5432/db";
    DatabaseConnection.instance = this;
  }
  
  query(sql) {
    console.log("Executing: " + sql);
  }
}

const db1 = new DatabaseConnection();
const db2 = new DatabaseConnection();
console.log(db1 === db2); // true (Identical instances)`
  },
  factory: {
    title: "Factory Method Pattern",
    desc: "Provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created. Decouples creation logic.",
    usecase: `// Factory Pattern for UI Button generation
class AndroidButton { render() { return "<android-button>"; } }
class iOSButton { render() { return "<ios-button>"; } }

class ButtonFactory {
  createButton(osType) {
    if (osType === 'android') return new AndroidButton();
    if (osType === 'ios') return new iOSButton();
    throw new Error("Unknown Operating System");
  }
}

const factory = new ButtonFactory();
const btn = factory.createButton('android');
console.log(btn.render()); // "<android-button>"`
  },
  observer: {
    title: "Observer Pattern",
    desc: "Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified automatically. Excellent for event brokers.",
    usecase: `// Newsletter subscription system
class NewsPublisher {
  constructor() { this.subscribers = []; }
  
  subscribe(observer) { this.subscribers.push(observer); }
  
  notify(news) {
    this.subscribers.forEach(sub => sub.update(news));
  }
}

class UserObserver {
  constructor(name) { this.name = name; }
  update(news) {
    console.log(this.name + " received news: " + news);
  }
}

const publisher = new NewsPublisher();
const aman = new UserObserver("Aman");
publisher.subscribe(aman);
publisher.notify("AI is evolving fast!");`
  },
  strategy: {
    title: "Strategy Pattern",
    desc: "Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.",
    usecase: `// Payment Processor Switcher
class PayPalStrategy {
  pay(amount) { console.log("PayPal Checkout: $" + amount); }
}

class StripeStrategy {
  pay(amount) { console.log("Stripe Credit Card: $" + amount); }
}

class CartContext {
  setPaymentMethod(strategy) { this.strategy = strategy; }
  checkout(amount) { this.strategy.pay(amount); }
}

const cart = new CartContext();
cart.setPaymentMethod(new StripeStrategy());
cart.checkout(99.00);`
  }
};

document.addEventListener('DOMContentLoaded', () => {
  // System Design Subtabs switcher
  const subTabBtns = document.querySelectorAll('.design-tab-btn');
  const subContents = document.querySelectorAll('.design-subcontent');
  
  subTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      subTabBtns.forEach(b => b.classList.remove('active'));
      subContents.forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      const target = btn.getAttribute('data-subtab');
      document.getElementById(target).classList.add('active');
      
      if (target === 'infra-simulator') {
        setTimeout(initInfraSimulator, 50);
      }
    });
  });

  // SOLID nav switcher
  const solidBtns = document.querySelectorAll('.solid-nav-btn');
  solidBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      solidBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadSOLIDDetails(btn.getAttribute('data-principle'));
    });
  });

  // Patterns switcher
  const patternItems = document.querySelectorAll('.pattern-item');
  patternItems.forEach(item => {
    item.addEventListener('click', () => {
      patternItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      loadPatternDetails(item.getAttribute('data-pattern'));
    });
  });

  // Default initial loads
  loadSOLIDDetails('s');
  loadPatternDetails('singleton');
});

function loadSOLIDDetails(key) {
  const data = window.solidPrinciples[key];
  const box = document.getElementById('solid-details');
  if (!data || !box) return;
  
  box.innerHTML = `
    <div class="solid-content-header">
      <h3>${data.title}</h3>
      <div class="desc">${data.desc}</div>
    </div>
    <div class="solid-content-body">
      <div class="code-compare-card dirty">
        <div class="compare-header"><i class="fa-solid fa-triangle-exclamation"></i> Non-Compliant Implementation</div>
        <div class="compare-code">${data.dirty}</div>
      </div>
      <div class="code-compare-card clean">
        <div class="compare-header"><i class="fa-solid fa-shield-halved"></i> SOLID Compliant Implementation</div>
        <div class="compare-code">${data.clean}</div>
      </div>
    </div>
  `;
}

function loadPatternDetails(key) {
  const data = window.designPatterns[key];
  const box = document.getElementById('pattern-details');
  if (!data || !box) return;
  
  box.innerHTML = `
    <h3>${data.title}</h3>
    <p class="subtitle" style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 8px;">${data.desc}</p>
    <div class="code-editor-card" style="border-radius:10px;">
      <div class="editor-header">Implementation Usecase</div>
      <div class="compare-code" style="padding: 1rem; background: rgba(0,0,0,0.25); font-family:'Fira Code',monospace; font-size:0.8rem; overflow-x:auto; white-space:pre;">${data.usecase}</div>
    </div>
  `;
}


// --- Infrastructure Scale & Cache Simulator ---
let infraCtx = null;
let infraCanvas = null;
let infraAnimationId = null;
let nodes = [];
let packets = [];
let isCacheEnabled = true;
let isServerBOnline = true;

function initInfraSimulator() {
  infraCanvas = document.getElementById('infra-canvas');
  if (!infraCanvas) return;
  infraCtx = infraCanvas.getContext('2d');
  
  // Set up nodes
  const W = infraCanvas.width;
  const H = infraCanvas.height;
  
  // Nodes layout
  nodes = {
    client: { x: 60, y: H / 2, label: "Client Browser", icon: "fa-laptop" },
    lb: { x: 200, y: H / 2, label: "Load Balancer", icon: "fa-code-fork" },
    srvA: { x: 360, y: H / 2 - 80, label: "Web Server A", icon: "fa-server", active: true },
    srvB: { x: 360, y: H / 2 + 80, label: "Web Server B", icon: "fa-server", active: true },
    redis: { x: 500, y: H / 2 - 80, label: "Redis Cache", icon: "fa-bolt", active: true },
    dbPri: { x: 620, y: H / 2 - 60, label: "Primary DB", icon: "fa-database" },
    dbRep: { x: 620, y: H / 2 + 60, label: "Replica DB", icon: "fa-copy" }
  };

  // Attach button triggers
  const btnRead = document.getElementById('btn-read-req');
  const btnWrite = document.getElementById('btn-write-req');
  const btnCache = document.getElementById('btn-toggle-cache');
  const btnServer = document.getElementById('btn-kill-server');
  
  // Clean event listeners to prevent duplicate stacking
  btnRead.replaceWith(btnRead.cloneNode(true));
  btnWrite.replaceWith(btnWrite.cloneNode(true));
  btnCache.replaceWith(btnCache.cloneNode(true));
  btnServer.replaceWith(btnServer.cloneNode(true));
  
  document.getElementById('btn-read-req').addEventListener('click', () => sendInfraPacket('read'));
  document.getElementById('btn-write-req').addEventListener('click', () => sendInfraPacket('write'));
  document.getElementById('btn-toggle-cache').addEventListener('click', toggleRedisCache);
  document.getElementById('btn-kill-server').addEventListener('click', toggleServerBStatus);
  
  // Reset packets
  packets = [];
  
  if (infraAnimationId) {
    cancelAnimationFrame(infraAnimationId);
  }
  tickInfraSimulator();
}

function sendInfraPacket(type) {
  // Determine route
  let targetServer = 'srvA';
  if (isServerBOnline) {
    // Round robin load balancing logic
    targetServer = Math.random() > 0.5 ? 'srvA' : 'srvB';
  }
  
  // Path list: sequence of node targets [nodeName, actionDelayDuration, resultMessage]
  let path = [];
  
  if (type === 'read') {
    path.push('client');
    path.push('lb');
    path.push(targetServer);
    
    if (isCacheEnabled) {
      path.push('redis'); // Try Cache check
      // 70% cache hit simulator rate
      const isHit = Math.random() > 0.3;
      if (isHit) {
        path.push(targetServer); // Return straight to Server
        path.push('lb');
        path.push('client');
      } else {
        // Cache miss -> Hit Replica DB
        path.push('dbRep');
        path.push('redis'); // Populate Cache
        path.push(targetServer);
        path.push('lb');
        path.push('client');
      }
    } else {
      // Direct database replica read
      path.push('dbRep');
      path.push(targetServer);
      path.push('lb');
      path.push('client');
    }
  } else {
    // Write request path (must hit primary DB)
    path.push('client');
    path.push('lb');
    path.push(targetServer);
    path.push('dbPri');
    // Replication event
    path.push('dbRep'); // Asynchronous replication visual
    
    // Write response flow returns early from dbPri
    path.push(targetServer);
    path.push('lb');
    path.push('client');
  }
  
  packets.push({
    type: type,
    path: path,
    currentIndex: 0,
    x: nodes.client.x,
    y: nodes.client.y,
    progress: 0, // travel progress between nodes (0 to 1)
    color: type === 'read' ? '#00e5ff' : '#a855f7',
    speed: 0.05
  });
}

function toggleRedisCache() {
  isCacheEnabled = !isCacheEnabled;
  const status = document.getElementById('cache-status');
  if (isCacheEnabled) {
    status.innerText = 'ENABLED';
    status.parentElement.className = 'btn btn-secondary';
  } else {
    status.innerText = 'DISABLED';
    status.parentElement.className = 'btn btn-danger';
  }
}

function toggleServerBStatus() {
  isServerBOnline = !isServerBOnline;
  const status = document.getElementById('server-status');
  if (isServerBOnline) {
    status.innerText = 'ONLINE';
    status.parentElement.className = 'btn btn-danger'; // Button toggles it off
    nodes.srvB.active = true;
  } else {
    status.innerText = 'OFFLINE';
    status.parentElement.className = 'btn btn-success'; // Button restarts it
    nodes.srvB.active = false;
  }
}

function tickInfraSimulator() {
  if (!infraCtx) return;
  
  const W = infraCanvas.width;
  const H = infraCanvas.height;
  
  infraCtx.clearRect(0, 0, W, H);
  
  // 1. Draw connections
  infraCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  infraCtx.lineWidth = 2;
  
  const drawLink = (n1, n2) => {
    infraCtx.beginPath();
    infraCtx.moveTo(nodes[n1].x, nodes[n1].y);
    infraCtx.lineTo(nodes[n2].x, nodes[n2].y);
    infraCtx.stroke();
  };
  
  drawLink('client', 'lb');
  if (nodes.srvA.active) drawLink('lb', 'srvA');
  if (nodes.srvB.active) drawLink('lb', 'srvB');
  drawLink('srvA', 'redis');
  drawLink('srvB', 'redis');
  drawLink('srvA', 'dbRep');
  drawLink('srvB', 'dbRep');
  drawLink('srvA', 'dbPri');
  drawLink('srvB', 'dbPri');
  drawLink('dbPri', 'dbRep');
  
  // 2. Render Nodes
  Object.keys(nodes).forEach(key => {
    const node = nodes[key];
    const isActive = key === 'srvB' ? isServerBOnline : true;
    
    // Glow nodes
    infraCtx.shadowColor = isActive ? (key === 'redis' ? '#f59e0b' : '#00e5ff') : '#ef4444';
    infraCtx.shadowBlur = isActive ? 8 : 4;
    
    infraCtx.fillStyle = isActive ? 'rgba(18, 22, 33, 0.85)' : 'rgba(239, 68, 68, 0.1)';
    infraCtx.strokeStyle = isActive ? 'rgba(255,255,255,0.1)' : '#ef4444';
    infraCtx.lineWidth = 1.5;
    
    // Draw rounded card
    const rw = 90, rh = 40;
    infraCtx.beginPath();
    infraCtx.roundRect(node.x - rw/2, node.y - rh/2, rw, rh, 8);
    infraCtx.fill();
    infraCtx.stroke();
    
    // Draw text
    infraCtx.shadowBlur = 0;
    infraCtx.fillStyle = isActive ? '#fff' : '#ef4444';
    infraCtx.font = 'bold 9px sans-serif';
    infraCtx.textAlign = 'center';
    infraCtx.fillText(node.label, node.x, node.y + 4);
    
    // Active/Offline indicator dots
    infraCtx.fillStyle = isActive ? '#10b981' : '#ef4444';
    infraCtx.beginPath();
    infraCtx.arc(node.x - rw/2 + 8, node.y - rh/2 + 8, 3, 0, 2*Math.PI);
    infraCtx.fill();
  });
  
  // 3. Update & Draw Packets
  packets.forEach((p, idx) => {
    const fromNode = nodes[p.path[p.currentIndex]];
    const toNode = nodes[p.path[p.currentIndex + 1]];
    
    if (toNode) {
      p.progress += p.speed;
      p.x = fromNode.x + (toNode.x - fromNode.x) * p.progress;
      p.y = fromNode.y + (toNode.y - fromNode.y) * p.progress;
      
      // Draw packet
      infraCtx.shadowColor = p.color;
      infraCtx.shadowBlur = 10;
      infraCtx.fillStyle = p.color;
      infraCtx.beginPath();
      infraCtx.arc(p.x, p.y, 5, 0, 2*Math.PI);
      infraCtx.fill();
      infraCtx.shadowBlur = 0;
      
      // Reach destination node
      if (p.progress >= 1) {
        p.currentIndex++;
        p.progress = 0;
        
        // Handle special asynchronous replication branch
        if (p.path[p.currentIndex] === 'dbPri' && p.path[p.currentIndex + 1] === 'dbRep') {
          // Fork a replication indicator packet in background
          packets.push({
            type: 'replication',
            path: ['dbPri', 'dbRep'],
            currentIndex: 0,
            x: nodes.dbPri.x,
            y: nodes.dbPri.y,
            progress: 0,
            color: '#10b981',
            speed: 0.03
          });
          
          // Re-route the primary write packet directly back to Web Server returning validation
          // The next element in its path array is already the return route
        }
      }
    } else {
      // Packet completed journey
      packets.splice(idx, 1);
      window.appState.addXP(2); // Earn small XP for interactive simulations!
    }
  });
  
  infraAnimationId = requestAnimationFrame(tickInfraSimulator);
}
