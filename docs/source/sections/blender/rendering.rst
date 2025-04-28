Rendering API
=============

Basic Example 
------------- 

Rendering an animation with a single blender instance can be done like so:

.. literalinclude:: ../../../../examples/blender/render_animation.py 
   :language: python   

| 

Parallelized Rendering
-----------------------

Making it use multiple instances is as easy as using :class:`BlenderClients <visionsim.simulate.blender.BlenderClients>` instead of :class:`BlenderClient <visionsim.simulate.blender.BlenderClient>` (notice the ``s``)! Here's all that needs to change: 

.. program-output:: git diff --word-diff=color --word-diff-regex=. --no-index --color ../../examples/blender/render_animation.py ../../examples/blender/render_animation_parallel.py | sed -z 's/.*@@.*@@....//g'
   :language: ansi-python-console
   :shell:

.. warning:: In practice, the number of rendering jobs will be limited by the user's system resources, most likely GPU VRAM. Start small, and increase accordingly.

|

Render Process Pool
------------------- 

As seen in above using multiple Blender instances to render a single scene is already faster than only using one, but it limits the user to rendering a single scene at a time. For more fine-grained control, you can use :meth:`BlenderClients.pool <visionsim.simulate.blender.BlenderClients.pool>`, which returns a `multiprocessing Pool <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool>`_-like instance which has had it's applicator methods (map/imap/starmap/etc) monkey-patched to inject a client instance as first argument:

.. literalinclude:: ../../../../examples/blender/renderpool_simple.py 
   :language: python 

This enables much more flexibility, allowing users to render multiple scenes or parts-thereof simultaneously. However, tracking their progress is not as simple as in the previous example as each render is now it's own process and needs to communicate it's state to the parent process. This is tricky to do correctly, thankfully, we can use the :class:`PoolProgress <visionsim.utils.progress.PoolProgress>` utility for this:

.. literalinclude:: ../../../../examples/blender/renderpool_progressbar.py 
   :language: python 

To properly track the progress, we queue render jobs with ``apply_async`` which gives control back to the main process. Then we call ``progress.wait()``, which blocks until all jobs have completed and updates the progress bar accordingly.    

.. important:: After queuing jobs, you *must* wait for them to terminate by calling :meth:`PoolProgress.wait <visionsim.utils.progress.PoolProgress.wait>`, otherwise the main process will exit and force-kill all child processes. 
