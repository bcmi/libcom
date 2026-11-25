(function () {
    function init() {
        const sidebar = document.querySelector("#pytorch-left-menu .pytorch-side-scroll");
        if (!sidebar) return;

        const scrollKey = "libcom-sidebar-scroll";
        const savedScroll = sessionStorage.getItem(scrollKey);
        let pendingScroll = savedScroll === null ? null : Number(savedScroll);
        sessionStorage.removeItem(scrollKey);

        function showCurrent() {
            const links = Array.from(sidebar.querySelectorAll("li.toctree-l1 > a"));
            const hash = window.location.hash;
            const match = hash && links.find((link) => {
                const url = new URL(link.href, window.location.href);
                return url.pathname === window.location.pathname && url.hash === hash;
            });

            if (match) {
                sidebar.querySelectorAll(".current").forEach((element) => {
                    element.classList.remove("current");
                });
                match.classList.add("current");
                match.parentElement.classList.add("current");
                match.parentElement.parentElement.classList.add("current");
            }

            const current = sidebar.querySelector("li.current > a");
            if (!current || sidebar.clientHeight === 0) return;

            if (pendingScroll !== null && sidebar.scrollHeight > sidebar.clientHeight) {
                sidebar.scrollTop = pendingScroll;
                pendingScroll = null;
            }

            const containerRect = sidebar.getBoundingClientRect();
            const itemRect = current.getBoundingClientRect();
            if (itemRect.top < containerRect.top) {
                sidebar.scrollTop += itemRect.top - containerRect.top;
            } else if (itemRect.bottom > containerRect.bottom) {
                sidebar.scrollTop += itemRect.bottom - containerRect.bottom;
            }
        }

        window.addEventListener("load", showCurrent);
        window.addEventListener("hashchange", showCurrent);
        window.addEventListener("popstate", showCurrent);
        if ("ResizeObserver" in window) {
            new ResizeObserver(showCurrent).observe(sidebar);
        }
        sidebar.addEventListener("click", (event) => {
            const link = event.target.closest("li.toctree-l1 > a");
            if (!link) return;
            const url = new URL(link.href, window.location.href);
            if (url.pathname !== window.location.pathname &&
                url.origin === window.location.origin &&
                event.button === 0 && !event.ctrlKey && !event.metaKey &&
                !event.shiftKey && !event.altKey && !link.target) {
                sessionStorage.setItem(scrollKey, sidebar.scrollTop);
            } else if (url.pathname === window.location.pathname && url.hash) {
                requestAnimationFrame(showCurrent);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
