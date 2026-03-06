/**
 * Navigation component for SSI-ENN BESS Valuation Platform (Private)
 */

const NAV_CONFIG = {
  brand: "SSI-ENN",
  brandSuffix: "BESS",
  version: "v4.0",
  logo_url: "https://ikenga.eu",
  pages: [
    { name: "Overview", path: "index.html" },
    { name: "Map Explorer", path: "map.html" },
    { name: "Regional", path: "regional.html" },
    { name: "Methodology", path: "methodology.html" },
    { name: "Data", path: "data.html" },
    { name: "Intelligence", path: "intelligence.html" }
  ],
  copyright: "Copyright © 2026 Altinium Invest S.r.L. All Rights Reserved.",
  disclaimer: "Proprietary & Confidential — For Authorised Subscribers Only"
};

function initNav() {
  renderTopNav();
  renderFooter();
  setActiveNavLink();
}

function renderTopNav() {
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';

  const navHTML = `
    <div class="topnav-content">
      <div class="topnav-brand">
        <a href="${NAV_CONFIG.logo_url}" target="_blank" class="topnav-brand-logo" title="Ikenga">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M3 7l9-4 9 4v10l-9 4-9-4V7z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
            <path d="M3 7l9 4m0 0l9-4m-9 4v10" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
          </svg>
        </a>
        <a href="index.html">
          ${NAV_CONFIG.brand} <span class="brand-suffix">${NAV_CONFIG.brandSuffix}</span>
          <span class="topnav-version">${NAV_CONFIG.version}</span>
        </a>
      </div>
      <ul class="topnav-menu">
        ${NAV_CONFIG.pages.map(page => `
          <li>
            <a href="${page.path}" class="${currentPath === page.path ? 'active' : ''}">${page.name}</a>
          </li>
        `).join('')}
      </ul>
    </div>
  `;

  const topnav = document.querySelector('.topnav');
  if (topnav) {
    topnav.innerHTML = navHTML;
  }
}

function renderFooter() {
  const footerHTML = `
    <p>${NAV_CONFIG.disclaimer}</p>
    <p>${NAV_CONFIG.copyright}</p>
  `;

  const footer = document.querySelector('.footer');
  if (footer) {
    footer.innerHTML = footerHTML;
  }
}

function setActiveNavLink() {
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.topnav-menu a').forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

// Initialize nav when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initNav);
} else {
  initNav();
}
