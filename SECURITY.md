# 🔒 SECURITY.md — Universal Vibe Coding Security Guide
> Drop this file into any project root and reference it with any AI agent:
> *"Before writing any code, read SECURITY.md and follow all rules strictly."*

---

## 🧠 HOW TO USE THIS FILE

Paste this instruction at the start of any AI coding session:

```
Before writing or modifying any code in this project, read SECURITY.md in the project root
and enforce every rule listed there. Flag any violation before proceeding. Never skip a rule
even if it seems unnecessary for the current task.
```

---

## 1. 🔑 SECRETS & ENVIRONMENT VARIABLES

**Rules:**
- NEVER hardcode API keys, passwords, tokens, or secrets in source code
- ALWAYS use `.env` files for secrets and load them via `dotenv` or equivalent
- ALWAYS add `.env` to `.gitignore` immediately when starting a project
- Provide a `.env.example` with placeholder values and NO real secrets
- Use different secrets for development, staging, and production environments
- Rotate any secret that was accidentally committed to git immediately

**Checklist for AI agents:**
```
[ ] No secrets in source files, config files, or comments
[ ] .env exists and is in .gitignore
[ ] .env.example exists with dummy values
[ ] No secrets in console.log / print statements
[ ] No secrets hardcoded in Docker files or CI/CD configs
```

**Bad:**
```js
const apiKey = "sk-abc123realkey"; // NEVER DO THIS
```

**Good:**
```js
const apiKey = process.env.API_KEY; // Always load from environment
if (!apiKey) throw new Error("API_KEY is not set in environment");
```

---

## 2. 🛡️ INPUT VALIDATION & SANITIZATION

**Rules:**
- NEVER trust user input — validate and sanitize everything on the server side
- Validate type, length, format, and range of all inputs
- Reject unexpected fields (use allowlists, not blocklists)
- Strip or encode HTML from any user-supplied strings before storing or rendering
- Use a validation library (e.g. `zod`, `joi`, `yup`, `pydantic`)

**Checklist:**
```
[ ] All API inputs validated with a schema
[ ] String length limits enforced
[ ] File uploads: type, size, and name validated
[ ] Numbers: min/max range enforced
[ ] Emails: format validated server-side
[ ] No raw user content inserted into HTML/SQL/shell commands
```

**Bad:**
```js
app.post('/user', (req, res) => {
  db.query(`INSERT INTO users VALUES ('${req.body.name}')`); // SQL injection!
});
```

**Good:**
```js
app.post('/user', (req, res) => {
  const { name } = userSchema.parse(req.body); // validated
  db.query('INSERT INTO users VALUES (?)', [name]); // parameterized
});
```

---

## 3. 💉 SQL INJECTION PREVENTION

**Rules:**
- ALWAYS use parameterized queries or prepared statements
- NEVER concatenate user input directly into SQL strings
- Use an ORM (Prisma, SQLAlchemy, TypeORM) when possible
- Limit database user permissions to only what the app needs (least privilege)

**Bad:**
```python
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

**Good:**
```python
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

---

## 4. 🕸️ XSS (CROSS-SITE SCRIPTING) PREVENTION

**Rules:**
- NEVER inject raw user content into the DOM via `innerHTML`, `dangerouslySetInnerHTML`, or `document.write`
- Always escape output in templates
- Use a Content Security Policy (CSP) header
- Sanitize rich text with a trusted library (e.g. `DOMPurify`)

**Bad:**
```js
document.getElementById('output').innerHTML = userInput; // XSS risk
```

**Good:**
```js
document.getElementById('output').textContent = userInput; // safe
// or for rich text:
element.innerHTML = DOMPurify.sanitize(userInput);
```

---

## 5. 🔐 AUTHENTICATION & AUTHORIZATION

**Rules:**
- NEVER roll your own auth system — use battle-tested libraries (NextAuth, Passport, Auth0, Supabase Auth)
- ALWAYS hash passwords with bcrypt, argon2, or scrypt — never MD5/SHA1
- Use JWT or session tokens with short expiry times
- Enforce authorization on EVERY protected route — never rely on frontend-only checks
- Implement proper logout (invalidate tokens server-side)
- Use HTTPS-only cookies with `HttpOnly` and `Secure` flags

**Checklist:**
```
[ ] Passwords hashed with bcrypt/argon2 (cost factor ≥ 12)
[ ] JWT secret is strong and stored in .env
[ ] Every protected API route checks auth token
[ ] Role-based access control enforced server-side
[ ] Sessions invalidated on logout
[ ] No sensitive data in JWT payload
```

---

## 6. 🚦 RATE LIMITING & BRUTE FORCE PROTECTION

**Rules:**
- Apply rate limiting to ALL public endpoints, especially auth routes
- Lock accounts or add delays after repeated failed login attempts
- Use `express-rate-limit`, `slowDown`, or equivalent middleware
- Add CAPTCHA to login/register forms for public-facing apps

**Example (Node.js):**
```js
import rateLimit from 'express-rate-limit';

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10, // max 10 attempts
  message: 'Too many attempts, please try again later'
});

app.use('/api/auth', authLimiter);
```

---

## 7. 🌐 CORS CONFIGURATION

**Rules:**
- NEVER use `CORS: *` (allow all) in production
- Explicitly whitelist allowed origins
- Only allow required HTTP methods
- Never reflect the `Origin` header blindly

**Bad:**
```js
app.use(cors()); // allows everyone
```

**Good:**
```js
app.use(cors({
  origin: ['https://yourdomain.com'],
  methods: ['GET', 'POST'],
  credentials: true
}));
```

---

## 8. 🪖 SECURITY HTTP HEADERS

**Rules:**
- Always set security headers in production
- Use `helmet` (Node.js) or equivalent for your stack
- Implement a Content Security Policy

**Minimum required headers:**
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
```

**Node.js one-liner:**
```js
import helmet from 'helmet';
app.use(helmet());
```

---

## 9. 📁 FILE UPLOAD SECURITY

**Rules:**
- Validate file type by MIME type AND extension (both can be spoofed — check both)
- Set a maximum file size limit
- NEVER store uploaded files in the web root / publicly accessible directory
- Rename uploaded files to random names (never use original filename)
- Scan uploads for malware if handling sensitive environments
- Never execute uploaded files

**Checklist:**
```
[ ] Allowed MIME types explicitly whitelisted
[ ] Max file size enforced (e.g. 5MB)
[ ] Files stored outside web root or in cloud storage (S3)
[ ] Filenames randomized (UUID)
[ ] No path traversal possible in filename
```

---

## 10. 🔓 DEPENDENCY SECURITY

**Rules:**
- Regularly audit dependencies with `npm audit`, `pip-audit`, or `snyk`
- Keep dependencies updated — outdated packages are the #1 attack vector
- Remove unused dependencies
- Lock dependency versions (`package-lock.json`, `requirements.txt`)
- Never install packages from untrusted or unofficial sources

**Commands to run regularly:**
```bash
npm audit fix          # Node.js
pip-audit              # Python
bundle audit           # Ruby
snyk test              # Universal
```

---

## 11. 🐛 ERROR HANDLING & INFORMATION LEAKAGE

**Rules:**
- NEVER expose stack traces, internal paths, database errors, or system info to users
- Return generic error messages to clients, log detailed errors server-side
- Never leak which specific field failed validation in auth errors (e.g. don't say "password incorrect" — say "invalid credentials")

**Bad:**
```js
app.use((err, req, res) => {
  res.json({ error: err.stack }); // exposes internals!
});
```

**Good:**
```js
app.use((err, req, res) => {
  console.error(err); // log internally
  res.status(500).json({ error: 'Something went wrong' }); // generic to user
});
```

---

## 12. 🔒 API KEY & THIRD-PARTY INTEGRATION SECURITY

**Rules:**
- Always use Read-Only API keys where full access isn't needed
- Store third-party API keys in `.env`, never in frontend code
- Never expose API keys in client-side JavaScript (they are public!)
- Use a backend proxy to make API calls on behalf of the frontend
- Set IP whitelisting on API keys when the provider supports it
- Regularly audit and rotate API keys

**Bad:**
```js
// In React/frontend code — ANYONE can see this in browser devtools
const response = await fetch('https://api.example.com', {
  headers: { 'Authorization': 'Bearer sk-realkey123' }
});
```

**Good:**
```js
// Frontend calls YOUR backend
const response = await fetch('/api/data');
// Your backend (server-side) holds the real API key
```

---

## 13. 🗄️ DATABASE SECURITY

**Rules:**
- Use least-privilege database users (app user should NOT have DROP/CREATE rights)
- Enable encryption at rest for sensitive databases
- Never expose the database port publicly — keep it on internal network only
- Regularly backup databases and test restoration
- Mask or encrypt PII fields (emails, phone numbers, SSNs)

---

## 14. 🚢 DEPLOYMENT & INFRASTRUCTURE

**Rules:**
- Always use HTTPS — never deploy over plain HTTP
- Use environment-specific configs (dev/staging/prod)
- Disable debug mode in production
- Remove development tools, test routes, and sample data before deploying
- Set up automated security scanning in CI/CD (GitHub Actions + `npm audit`)
- Use a Web Application Firewall (WAF) for public-facing apps

**Checklist before every deployment:**
```
[ ] DEBUG=false in production
[ ] No test/dev routes exposed
[ ] All secrets in environment variables, not code
[ ] HTTPS enabled
[ ] Database not publicly accessible
[ ] npm audit / pip-audit passes with no critical issues
[ ] Security headers configured
[ ] Rate limiting enabled
```

---

## 15. 📝 LOGGING & MONITORING

**Rules:**
- Log authentication events (login, logout, failed attempts)
- Log all admin actions
- NEVER log passwords, tokens, or sensitive personal data
- Set up alerts for anomalous activity (many failed logins, unusual data access)
- Retain logs in a separate, secure location

---

## 🚨 IMMEDIATE RED FLAGS — AI AGENT MUST STOP AND WARN

If any AI agent encounters any of the following while coding, it **must stop and flag it**:

```
🔴 API key or secret found in source code
🔴 SQL query built with string concatenation from user input
🔴 innerHTML used with user-controlled data
🔴 Password stored in plaintext or hashed with MD5/SHA1
🔴 CORS set to wildcard (*) in production config
🔴 .env file not in .gitignore
🔴 Auth check missing on a protected route
🔴 Stack trace or internal error returned to user
🔴 User-supplied filename used directly for file storage
🔴 No rate limiting on login/register endpoints
```

---

## 📋 QUICK-START SECURITY CHECKLIST (per project)

```
[ ] .env created and added to .gitignore
[ ] .env.example created with placeholder values
[ ] Input validation library installed and used
[ ] Parameterized queries used everywhere
[ ] Auth library chosen (no custom auth)
[ ] Passwords hashed with bcrypt/argon2
[ ] Security headers configured (helmet or equivalent)
[ ] CORS explicitly configured
[ ] Rate limiting on auth routes
[ ] Error handler returns generic messages
[ ] npm audit / pip-audit clean
[ ] HTTPS configured for deployment
[ ] All API keys are server-side only
[ ] File upload security implemented (if applicable)
[ ] Database user has minimal permissions
```

---

*Last updated: 2026 — Compatible with Node.js, Python, React, Next.js, and general web projects.*
*Reference this file at the start of every AI coding session.*
