# 🔑 הגדרת API Keys לנתונים פיננסיים אמיתיים

## 📊 Alpha Vantage (מומלץ)
1. הרשמה חינמית: https://www.alphavantage.co/support/#api-key
2. תקבל key דמוי: `ABC123XYZ789`
3. הוספה ל-Docker environment:

### Windows PowerShell:
```powershell
# הוספה זמנית לסשן
$env:ALPHA_VANTAGE_API_KEY = "ABC123XYZ789"

# או עדכון קובץ .env
echo "ALPHA_VANTAGE_API_KEY=ABC123XYZ789" >> .env
```

### לינוקס/מק:
```bash
export ALPHA_VANTAGE_API_KEY="ABC123XYZ789"
echo "ALPHA_VANTAGE_API_KEY=ABC123XYZ789" >> .env
```

## 🔄 הפעלה עם API Key:
```bash
docker-compose down
docker-compose up -d
```

## 📈 Yahoo Finance (אלטרנטיבה)
- לא דורש API key
- לפעמים נחסם או לא יציב
- המערכת מנסה אותו אוטומטית תחילה

## 🎯 בדיקת מקור הנתונים:
```bash
# הצגת לוגים לבדיקת מקור הנתונים
docker-compose logs api | grep -i "yahoo\|alpha\|demo"
```

## ✅ סטטוס הנוכחי:
- **Yahoo Finance**: נכשל (בעיות תקשורת)
- **Alpha Vantage**: לא מוגדר (אין API key)  
- **נתוני דמו**: פעיל ומציאותי ✓

## 💡 המלצה:
המערכת כרגע עובדת מצוין עם נתוני דמו ריאליסטיים.
לייצור - מומלץ להשיג Alpha Vantage key חינמי.

---

## 🤖 Gemini (לשימושי LLM עתידיים/העשרה)

אנחנו תומכים בזיהוי מפתח של Gemini לצורך העשרה עתידית של תובנות (ללא נתוני דמו). כרגע המפתח נבדק דרך בריאות ספקים, ואפשר להגדיר אותו כמשתנה סביבה:

### Windows PowerShell:
```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"
```

### Linux/Mac:
```bash
export GEMINI_API_KEY="YOUR_GEMINI_KEY_HERE"
```

לאחר ההגדרה, אפשר לוודא שהמערכת רואה את המפתח:

```powershell
# בדיקת זמינות ספקים (כולל Gemini)
Invoke-RestMethod -Uri "http://localhost:8000/api/sentiment/providers" | ConvertTo-Json -Depth 5
```

הערה: המערכת עובדת במצב Live-only; אם נתוני סנטימנט אינם זמינים בפועל מהספקים (NewsAPI/Bing/Alpha Vantage/Yahoo), תחזור תשובת 200 עם הודעה ידידותית "No sentiment data available" – ללא נתוני דמה.