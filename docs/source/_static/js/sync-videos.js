/* Play the clips of a scene in phase.

   Browsers start a clip once that particular file has buffered enough, so renders of
   the same scene that load a moment apart drift apart and the only thing moving on the
   page is the difference between them. Once every clip of a group can play, seeking
   them all back to zero in one loop puts them on the same clock. */

(() => {
  "use strict";

  const clipsOnPage = () => Array.from(document.querySelectorAll("video"));

  /** Clips that share a container: figures of one scene sit in the same grid or section. */
  const containerOf = (clip, clips) => {
    let node = clip.parentElement;
    while (node && node !== document.body) {
      if (clips.filter((other) => node.contains(other)).length > 1) return node;
      node = node.parentElement;
    }
    // Nothing narrower holds more than one clip, so they all belong together.
    return document.body;
  };

  const groupByContainer = (clips) => {
    const groups = new Map();
    for (const clip of clips) {
      const container = containerOf(clip, clips);
      if (!groups.has(container)) groups.set(container, []);
      groups.get(container).push(clip);
    }
    return groups;
  };

  /** Resolve once the browser has a frame to restart from. */
  const playable = (clip) =>
    clip.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
      ? Promise.resolve()
      : new Promise((resolve) => {
          clip.addEventListener("loadeddata", resolve, { once: true });
          clip.addEventListener("error", resolve, { once: true });
        });

  const sync = (clips) =>
    Promise.all(clips.map(playable)).then(() => {
      for (const clip of clips) {
        // One loop, so every clip restarts on the same tick. Autoplay can still be
        // refused, which leaves a clip parked on its poster frame: not an error here.
        clip.currentTime = 0;
        clip.play().catch(() => {});
      }
    });

  // A group is put in phase once. Syncing again on every `loadeddata` would seek clips
  // that are already playing together, which reads as a stutter.
  const synced = new Set();

  const syncGroups = () => {
    const clips = clipsOnPage();
    if (clips.length < 2) return;
    for (const [container, group] of groupByContainer(clips)) {
      if (group.length < 2 || synced.has(container)) continue;
      synced.add(container);
      sync(group);
    }
  };

  const start = () => {
    // `loadeddata` does not bubble, but it is still delivered on the way down, so this
    // also catches a clip that a page script adds after load.
    document.addEventListener("loadeddata", syncGroups, true);
    syncGroups();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
