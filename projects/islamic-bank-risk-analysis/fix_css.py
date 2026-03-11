
import os

style_path = r'C:\Users\Jumanazar\Desktop\islamic-bank-risk-analysis\frontend\css\style.css'

with open(style_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# We want to reconstruct the file.
# The original file has parts we want to keep, parts to insert, and parts to discard.

# 1-275: Keep mostly (contains :root, base, sidebar, top-header)
# After 275: Insert .main-content
# Keep the rest until ~770 (cards, charts, etc)
# After 770: Discard the messy !important stuff and append the clean footer/responsive.

clean_tail = """
/* ── USER PROFILE ── */
.user-profile-box {
  margin-top: auto;
  padding: 16px;
  border-top: 1px solid var(--border-default);
}

.user-profile-content {
  display: flex;
  align-items: center;
  gap: 12px;
  background: var(--bg-elevated);
  padding: 10px 14px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-sm);
}

.user-avatar {
  width: 36px;
  height: 36px;
  background: var(--brand-primary);
  color: var(--text-inverse);
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 15px;
  flex-shrink: 0;
}

.user-info {
  flex: 1;
  min-width: 0;
}

.user-name {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.user-role {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600;
}

.logout-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 8px;
  border-radius: var(--radius-md);
  transition: var(--transition);
}

.logout-btn:hover {
  background: rgba(239, 68, 68, 0.1);
  color: #EF4444;
}

.sidebar-footer {
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 11px;
  color: var(--text-muted);
}

/* ── RESPONSIVE ── */
@media (max-width: 1280px) {
  :root { --sidebar-w: 240px; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 768px) {
  .sidebar { transform: translateX(-100%); }
  .sidebar.open { transform: translateX(0); }
  .main-content { margin-left: 0; padding-left: 16px; padding-right: 16px; }
  .top-header { left: 0; padding: 0 16px; }
}
"""

main_content_def = """
/* ── MAIN CONTENT ── */
.main-content {
  margin-left: var(--sidebar-w);
  padding-top: var(--header-h);
  padding-left: 32px;
  padding-right: 32px;
  padding-bottom: 32px;
  min-height: 100vh;
  flex: 1;
  transition: var(--transition);
}
"""

# Rebuild process
new_content = []

# Part 1: up to .header-title
header_title_block_found = False
for i, line in enumerate(lines):
    new_content.append(line)
    if ".header-title {" in line:
        # found it, wait for the closing brace
        pass
    if i > 250 and "color: var(--text-primary);" in line and "}" in lines[i+1]:
        # this is likely the end of header-title
        new_content.append(lines[i+1])
        new_content.append(main_content_def)
        header_title_block_found = True
        skip_to = i + 2
        break

if not header_title_block_found:
    # fallback to just line 275
    new_content = lines[:275]
    new_content.append(main_content_def)
    skip_to = 275

# Part 2: Middle cards/charts/etc
# We keep until we see the "USER PROFILE" messy block or ~770
for i in range(skip_to, len(lines)):
    if "/*  USER PROFILE  */" in lines[i] or i > 770:
        break
    new_content.append(lines[i])

# Part 3: Append clean tail
new_content.append(clean_tail)

final_text = "".join(new_content)

with open(style_path, 'w', encoding='utf-8') as f:
    f.write(final_text)

print("SUCCESS: style.css rebuilt safely.")
