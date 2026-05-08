// FitnessSystem Core App
const COACHES = {
  זכרים: {
    'אלכס (35)': { אופי: 'צבאי/סמכותי', סגנון: 'פקודות ישירות' },
    'ליאו (29)': { אופי: 'אנליטי/מדעי', סגנון: 'הסברים מדעיים' },
    'בן (26)': { אופי: 'חברי/משוחרר', סגנון: 'שיחה זורמת' },
    'דיוויד (45)': { אופי: 'חכם/זן', סגנון: 'מדיטטיבי' },
    'ויקטור (30)': { אופי: 'תחרותי/אגרסיבי', סגנון: 'דחיפה לקצה' },
  },
  נקבות: {
    'מיה (34)': { אופי: 'מכילה/תומכת', סגנון: 'עידוד מתמיד' },
    'זואי (24)': { אופי: 'אנרגטית/הייפ', סגנון: 'קצבי וצעקני' },
    'נועה (31)': { אופי: 'הוליסטית/מאוזנת', סגנון: 'זרימה' },
    'טלי (38)': { אופי: 'עסקית/חדה', סגנון: 'תמציתי' },
    'סיון (25)': { אופי: 'מומחית שינוי', סגנון: 'פסיכולוגי' },
  },
};

class FitnessSystem {
  constructor() {
    this.state = 'home';
    this.userData = null;
    this.formData = {
      name: '',
      age: '',
      weight: '',
      height: '',
      gender: '',
      goal: '',
    };
    this.selectedGender = null;
    this.selectedCoach = null;
    this.bmr = null;
    this.init();
  }

  init() {
    this.loadUserData();
    this.renderHome();
  }

  calculateBMR(weight, height, age, gender) {
    if (gender === 'זכר') {
      return 88.362 + 13.397 * weight + 4.799 * height - 5.677 * age;
    } else {
      return 447.593 + 9.247 * weight + 3.098 * height - 4.33 * age;
    }
  }

  loadUserData() {
    const saved = localStorage.getItem('fitnessUserData');
    if (saved) {
      this.userData = JSON.parse(saved);
    }
  }

  saveUserData() {
    if (this.userData) {
      localStorage.setItem('fitnessUserData', JSON.stringify(this.userData));
    }
  }

  renderHome() {
    document.body.innerHTML = `
      <div class="app-container">
        <header class="header">
          <h1>💪 מערכת כושר</h1>
          <p>v1.0.0</p>
        </header>

        <div class="content">
          <div class="card">
            <h2>ברוכים הבאים</h2>
            <p>בואו נתחילו בתהליך אונבורדינג אישי עם מאמן שבחרת</p>
          </div>

          <button class="btn btn-primary" onclick="fitnessApp.renderOnboarding()">
            התחל אונבורדינג
          </button>

          ${this.userData ? `
            <button class="btn btn-secondary" onclick="fitnessApp.renderResults()">
              הצג פרופיל שלי
            </button>
          ` : ''}

          <div class="features">
            <h3>תכונות</h3>
            <div class="feature">✓ 10 מאמנים עם אישיויות שונות</div>
            <div class="feature">✓ חישוב BMR מדויק</div>
            <div class="feature">✓ מעקב התשובות שלך</div>
            <div class="feature">✓ סנכרון בין מכשירים</div>
          </div>
        </div>
      </div>
    `;
  }

  renderOnboarding() {
    let html = `
      <div class="app-container">
        <header class="header">
          <button onclick="fitnessApp.renderHome()" class="back-btn">← חזור</button>
          <h1>פרטים אישיים</h1>
        </header>

        <div class="content">
          <form id="onboardingForm">
            <div class="form-group">
              <label>שם מלא</label>
              <input type="text" id="name" placeholder="שם מלא" value="${this.formData.name}">
            </div>

            <div class="form-group">
              <label>גיל</label>
              <input type="number" id="age" placeholder="גיל" value="${this.formData.age}">
            </div>

            <div class="form-group">
              <label>משקל (ק"ג)</label>
              <input type="number" id="weight" placeholder="משקל" value="${this.formData.weight}">
            </div>

            <div class="form-group">
              <label>גובה (ס"מ)</label>
              <input type="number" id="height" placeholder="גובה" value="${this.formData.height}">
            </div>

            <div class="form-group">
              <label>מין</label>
              <div class="gender-buttons">
                <button type="button" class="btn ${this.formData.gender === 'זכר' ? 'active' : ''}"
                  onclick="fitnessApp.selectGender('זכר')">זכר</button>
                <button type="button" class="btn ${this.formData.gender === 'נקבה' ? 'active' : ''}"
                  onclick="fitnessApp.selectGender('נקבה')">נקבה</button>
              </div>
            </div>

            <div class="form-group">
              <label>מטרה</label>
              <input type="text" id="goal" placeholder="מטרה (ירידה משקל/שריר/כללית)" value="${this.formData.goal}">
            </div>
    `;

    if (this.selectedGender) {
      html += `
        <div class="form-group">
          <label>בחר מאמן</label>
          <div class="coaches-list">
      `;
      for (const [coachName, coach] of Object.entries(COACHES[this.selectedGender])) {
        html += `
          <button type="button" class="coach-card ${this.selectedCoach === coachName ? 'active' : ''}"
            onclick="fitnessApp.selectCoach('${coachName}')">
            <div class="coach-name">${coachName}</div>
            <div class="coach-style">${coach.סגנון}</div>
          </button>
        `;
      }
      html += `
          </div>
        </div>
      `;
    }

    html += `
            <button type="button" class="btn btn-primary" onclick="fitnessApp.handleOnboarding()">
              סיים אונבורדינג
            </button>
          </form>
        </div>
      </div>
    `;

    document.body.innerHTML = html;
  }

  selectGender(gender) {
    this.formData.gender = gender;
    this.selectedGender = gender === 'זכר' ? 'זכרים' : 'נקבות';
    this.renderOnboarding();
  }

  selectCoach(coachName) {
    this.selectedCoach = coachName;
    this.renderOnboarding();
  }

  handleOnboarding() {
    this.formData.name = document.getElementById('name')?.value || '';
    this.formData.age = parseInt(document.getElementById('age')?.value || 0);
    this.formData.weight = parseFloat(document.getElementById('weight')?.value || 0);
    this.formData.height = parseInt(document.getElementById('height')?.value || 0);
    this.formData.goal = document.getElementById('goal')?.value || '';

    if (!this.formData.name || !this.formData.age || !this.formData.weight ||
        !this.formData.height || !this.formData.gender || !this.formData.goal || !this.selectedCoach) {
      alert('אנא מלא את כל השדות');
      return;
    }

    const bmr = this.calculateBMR(this.formData.weight, this.formData.height, this.formData.age, this.formData.gender);
    const coach = COACHES[this.selectedGender][this.selectedCoach];

    this.userData = {
      ...this.formData,
      coach: this.selectedCoach,
      coachStyle: coach.סגנון,
      bmr: Math.round(bmr),
      timestamp: new Date().toLocaleString('he-IL'),
    };

    this.saveUserData();
    this.renderResults();
  }

  renderResults() {
    if (!this.userData) {
      this.renderHome();
      return;
    }

    document.body.innerHTML = `
      <div class="app-container">
        <header class="header">
          <button onclick="fitnessApp.renderHome()" class="back-btn">← תחילה</button>
          <h1>🎉 מברוק!</h1>
        </header>

        <div class="content">
          <div class="card">
            <div class="result-item">
              <span class="label">שם</span>
              <span class="value">${this.userData.name}</span>
            </div>
            <div class="result-item">
              <span class="label">גיל</span>
              <span class="value">${this.userData.age}</span>
            </div>
            <div class="result-item">
              <span class="label">משקל (ק"ג)</span>
              <span class="value">${this.userData.weight}</span>
            </div>
            <div class="result-item">
              <span class="label">גובה (ס"מ)</span>
              <span class="value">${this.userData.height}</span>
            </div>
            <div class="result-item">
              <span class="label">מטרה</span>
              <span class="value">${this.userData.goal}</span>
            </div>
            <div class="result-item">
              <span class="label">מאמן</span>
              <span class="value">${this.userData.coach}</span>
            </div>
            <div class="result-item">
              <span class="label">סגנון</span>
              <span class="value">${this.userData.coachStyle}</span>
            </div>
            <div class="result-item">
              <span class="label">BMR (קלוריות/יום)</span>
              <span class="value">${this.userData.bmr}</span>
            </div>
          </div>

          <div class="message-box">
            <p>הנתונים שלך נשמרו בהצלחה! המאמן שלך מוכן לעזור לך.</p>
          </div>

          <button class="btn btn-primary" onclick="fitnessApp.reset()">
            התחל מחדש
          </button>
        </div>
      </div>
    `;
  }

  reset() {
    this.userData = null;
    this.formData = {
      name: '',
      age: '',
      weight: '',
      height: '',
      gender: '',
      goal: '',
    };
    this.selectedGender = null;
    this.selectedCoach = null;
    localStorage.removeItem('fitnessUserData');
    this.renderHome();
  }
}

// Initialize app
const fitnessApp = new FitnessSystem();
