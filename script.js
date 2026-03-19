// Tab navigation via URL hash
function activate() {
  var hash = location.hash || '#overview';
  var regions = document.querySelectorAll('.content-region');
  var links = document.querySelectorAll('.main-menu a');

  regions.forEach(function(el) { el.classList.add('hide'); });
  links.forEach(function(el) { el.classList.remove('active'); });

  var target = document.querySelector(hash);
  if (target) { target.classList.remove('hide'); }

  var activeLink = document.querySelector('.main-menu a[href="' + hash + '"]');
  if (activeLink) { activeLink.classList.add('active'); }
}

window.addEventListener('load', activate);
window.addEventListener('hashchange', activate);
