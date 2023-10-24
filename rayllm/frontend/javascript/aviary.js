// Set favicon
const FAVICON =
  "data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>";
function setFavicon(link) {
  let favicon = document.querySelector('link[rel="icon"]');

  if (favicon) {
    favicon.href = link;
  } else {
    favicon = document.createElement("link");
    favicon.rel = "icon";
    favicon.href = link;

    document.head.appendChild(favicon);
  }
}
setFavicon(FAVICON);

// Get news
const NEWS_URL = "https://api.github.com/repos/ray-project/aviary/issues/8";
function getNews(newsUrl) {
  return fetch(newsUrl)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Unable to fetch news.");
      }
      return response.text();
    })
    .then((data) => {
      return (title = JSON.parse(data)["title"]);
    })
    .catch((error) => console.error("Unable to parse response: ", error));
}

// Wait for the ticker div to be added to DOM to set the news content
const observer = new MutationObserver((mutationsList, observer) => {
  for (let mutation of mutationsList) {
    if (mutation.type === "childList") {
      let element = document.getElementsByClassName("ticker");
      if (element.length > 0) {
        getNews(NEWS_URL).then((newsTitle) => {
          document.getElementsByClassName("ticker")[0].innerHTML =
            "\uD83D\uDCE3 " + newsTitle;
        });
        observer.disconnect(); 
        break;
      }
    }
  }
});

(function () {
  // Add Google Tag Manager
  const head = document.getElementsByTagName("head")[0];
  var gtm = document.createElement("script");
  gtm.text =
    "(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','GTM-5ZPDX2P');";
  head.insertBefore(gtm, head.children[0]);

  document.addEventListener("DOMContentLoaded", function () {
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();