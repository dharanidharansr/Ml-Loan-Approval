# 🔐 GitHub Secrets Configuration Guide

## Required Secrets for CI/CD Pipeline

To make your CI/CD pipeline fully functional, you need to configure the following secrets in your GitHub repository.

### 🛠️ How to Add Secrets

1. Go to your GitHub repository: `https://github.com/dharanidharansr/Ml-Loan-Approval`
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **"New repository secret"** button
5. Add each secret from the list below

---

## 📋 Required Secrets

### 1. **RENDER_DEPLOY_HOOK_URL** (Required for Deployment)

**What it is**: URL that triggers automatic deployment on Render
**Where to get it**: 
1. Go to [render.com](https://render.com) → Your Dashboard
2. Select your loan approval service
3. Go to **Settings** → **Build & Deploy**
4. Copy the **"Deploy Hook"** URL

**Format**: `https://api.render.com/deploy/srv-xxxxxxxxx?key=yyyyyyyyy`

**Secret Name**: `RENDER_DEPLOY_HOOK_URL`
**Secret Value**: (paste the entire URL from Render)

### 2. **APP_URL** (Required for Health Checks)

**What it is**: Your deployed application URL for health checks
**Where to get it**: Your Render service URL

**Format**: `https://your-app-name.onrender.com`

**Secret Name**: `APP_URL`
**Secret Value**: (your full Render app URL)

---

## 🔔 Optional Secrets (for Notifications)

### 3. **SLACK_WEBHOOK_URL** (Optional - for Slack notifications)

**What it is**: Slack webhook URL for deployment notifications
**How to create**:
1. Go to your Slack workspace
2. Create a new app at [api.slack.com](https://api.slack.com/apps)
3. Enable Incoming Webhooks
4. Create a webhook for your desired channel
5. Copy the webhook URL

**Format**: `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX`

**Secret Name**: `SLACK_WEBHOOK_URL`
**Secret Value**: (your Slack webhook URL)

### 4. **SNYK_TOKEN** (Optional - for security scanning)

**What it is**: Snyk token for vulnerability scanning
**How to get**:
1. Sign up at [snyk.io](https://snyk.io)
2. Go to Account Settings → API Token
3. Copy your token

**Secret Name**: `SNYK_TOKEN`
**Secret Value**: (your Snyk API token)

---

## 🚀 Step-by-Step Setup

### Step 1: Get Render Deploy Hook

```bash
# 1. Login to Render Dashboard
# 2. Navigate to: Your Service → Settings → Build & Deploy
# 3. Find "Deploy Hook" section
# 4. Copy the URL (looks like):
#    https://api.render.com/deploy/srv-abc123def456?key=xyz789uvw012
```

### Step 2: Add to GitHub Secrets

```bash
# 1. Go to GitHub repository
# 2. Settings → Secrets and variables → Actions
# 3. Click "New repository secret"
# 4. Add:
#    Name: RENDER_DEPLOY_HOOK_URL
#    Value: [paste the Render deploy hook URL]
```

### Step 3: Add App URL

```bash
# 1. Get your Render app URL (usually shown in dashboard)
# 2. Add secret:
#    Name: APP_URL  
#    Value: https://your-app-name.onrender.com
```

### Step 4: Test the Pipeline

```bash
# Make a small change and push to main branch
echo "# Test" >> README.md
git add README.md
git commit -m "Test CI/CD pipeline"
git push origin main

# Check GitHub Actions tab for workflow execution
```

---

## 🔍 Verification

After adding secrets, your GitHub Secrets page should show:

```
Repository secrets:
✅ RENDER_DEPLOY_HOOK_URL
✅ APP_URL
🔔 SLACK_WEBHOOK_URL (optional)
🔒 SNYK_TOKEN (optional)
```

---

## 🛠️ What Each Secret Does

| Secret | Purpose | Required | What happens if missing |
|--------|---------|----------|------------------------|
| `RENDER_DEPLOY_HOOK_URL` | Triggers Render deployment | ✅ Yes | Deployment will be skipped with warning |
| `APP_URL` | Health check after deployment | ✅ Yes | Health check will be skipped with warning |
| `SLACK_WEBHOOK_URL` | Send notifications to Slack | ❌ No | Notifications will be skipped silently |
| `SNYK_TOKEN` | Security vulnerability scanning | ❌ No | Security scan will be skipped |

---

## 🐛 Troubleshooting

### "Error: Specify secrets.SLACK_WEBHOOK_URL"
- **Cause**: Slack webhook secret not configured
- **Solution**: Either add the secret or remove Slack notification steps

### "curl: (7) Failed to connect"
- **Cause**: Wrong `RENDER_DEPLOY_HOOK_URL` or `APP_URL`
- **Solution**: Double-check the URLs in Render dashboard

### "Deployment skipped with warning"
- **Cause**: Missing `RENDER_DEPLOY_HOOK_URL` secret
- **Solution**: Add the secret following Step 1-2 above

### "Health check skipped with warning"
- **Cause**: Missing `APP_URL` secret
- **Solution**: Add your Render app URL as `APP_URL` secret

---

## 📊 Pipeline Behavior

With secrets configured:
```
✅ Tests run
✅ Code quality checks
✅ Security scanning (if SNYK_TOKEN provided)
✅ Deployment to Render (if RENDER_DEPLOY_HOOK_URL provided)
✅ Health check (if APP_URL provided)  
✅ Slack notifications (if SLACK_WEBHOOK_URL provided)
```

Without secrets:
```
✅ Tests run
✅ Code quality checks
⚠️ Security scanning skipped
⚠️ Deployment skipped with warning
⚠️ Health check skipped with warning
⚠️ Notifications skipped
```

The pipeline will continue to work even without secrets, but deployment and notifications will be skipped.

---

## 🔐 Security Best Practices

1. **Never commit secrets** to your repository
2. **Use environment-specific secrets** for staging vs production
3. **Rotate secrets regularly** (especially webhook URLs)
4. **Limit secret access** to necessary team members only
5. **Monitor secret usage** in GitHub Actions logs

---

## 📞 Need Help?

1. Check the **TROUBLESHOOTING.md** file for common issues
2. Review GitHub Actions workflow logs for detailed error messages
3. Verify all secrets are correctly formatted (no extra spaces, complete URLs)
4. Test each component individually using the local test script: `./test_local.sh`

Once you add the required secrets (`RENDER_DEPLOY_HOOK_URL` and `APP_URL`), your CI/CD pipeline will be fully functional! 🎉
