/* Interactive node graphs.

   Each `.node-graph` element carries the Tree Clipper export it draws as an inline
   `data-payload` (the `node-graph` directive puts it there, compressed), so nothing
   has to be fetched and the graphs also work when the docs are opened from disk.
   The viewer is a React app pulled from a CDN, so it is only imported on pages that
   actually have a graph, and its stylesheet is only added there too. The graph
   renders dark in both docs themes, which is how Blender's node editor looks. */

(() => {
  "use strict";

  const CDN = "https://esm.sh/geonodes-web-render@0.3";
  const selector = ".node-graph[data-payload]";

  /* A hand-drawn monitor glyph: the viewer's own controls already use corner brackets
     and arrows, so the toggle needs a shape of its own. The label lives in the tooltip. */
  const ICON =
    '<svg viewBox="0 0 16 16" width="15" height="15" fill="currentColor" aria-hidden="true">' +
    '<path d="M1.2 2.2h13.6v8.2H9.9l1 2.1h1.6v1.3H3.5v-1.3h1.6l1-2.1H1.2V2.2zM2.6 3.6V9h10.8V3.6H2.6z"/></svg>';

  const fail = (elements) => {
    for (const element of elements) {
      element.classList.add("node-graph--failed");
      element.textContent =
        "Graph unavailable (the viewer is loaded from a CDN). " +
        "Save the blend file to inspect the nodes in Blender.";
    }
  };

  /* Ask the viewer to fit its graph to the container. Its own control is the only fit
     entry point it exposes, so click that rather than reaching into its canvas. */
  const refit = (shell) => {
    const control = shell.querySelector(".react-flow__controls-fitview");
    // Let the container settle back to its inline size first, or the fit is computed
    // against the full screen viewport.
    if (control) window.setTimeout(() => control.click(), 80);
  };

  /* The toggle lives next to the viewer's container rather than inside it: React owns
     that container, and a sibling survives the graph being replaced - and stays
     visible when the wrapper is what goes full screen. */
  const addFullscreenToggle = (shell) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "node-graph__fullscreen";
    button.innerHTML = ICON;
    button.title = "Show the graph full screen";
    button.setAttribute("aria-pressed", "false");

    let full = false;
    const setState = (next) => {
      full = next;
      button.title = next ? "Exit full screen" : "Show the graph full screen";
      button.setAttribute("aria-pressed", String(next));
    };

    button.addEventListener("click", () => {
      if (document.fullscreenElement === shell) {
        document.exitFullscreen();
      } else if (shell.requestFullscreen) {
        shell.requestFullscreen();
      }
    });

    // Escape and the browser's own controls both leave full screen, so follow the event
    // rather than tracking it in the click handler. Every graph gets this listener, so
    // only the one whose own state changed may react.
    document.addEventListener("fullscreenchange", () => {
      const next = document.fullscreenElement === shell;
      if (next === full) return;
      setState(next);
      if (!next) refit(shell);
    });
    shell.append(button);
  };

  const wrap = (element) => {
    const shell = document.createElement("div");
    shell.className = "node-graph-shell";
    element.before(shell);
    shell.append(element);
    addFullscreenToggle(shell);
    return shell;
  };

  const start = async () => {
    const elements = Array.from(document.querySelectorAll(selector));
    if (elements.length === 0) return;

    for (const element of elements) wrap(element);

    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = `${CDN}/dist/embed.css`;
    document.head.append(stylesheet);

    try {
      const { mountGraphView } = await import(`${CDN}/embed`);
      for (const element of elements) {
        mountGraphView(element, {
          payload: element.dataset.payload,
          allowSelection: false,
        });
      }
    } catch (error) {
      fail(elements);
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
