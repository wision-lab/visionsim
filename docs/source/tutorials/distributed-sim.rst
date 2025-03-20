Distributed Simulation
======================

As seen in :doc:`parallel-sim` using multiple Blender instances to render a single scene is already faster than only using one, but it limits the user to rendering a single scene at a time. For more fine-grained control, you can use :meth:`BlenderClients.pool <spsim.simulate.blender.BlenderClients.pool>`, which returns a `multiprocessing Pool <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool>`_-like instance which has had it's applicator methods (map/imap/starmap/etc) monkey-patched to inject a client instance as first argument:

.. literalinclude:: ../../../examples/renderpool_simple.py 
   :language: python 

This enables much more flexibility, allowing users to render multiple scenes or parts-thereof simultaneously. However, tracking their progress is not as simple as in the previous example as each render is now it's own process and needs to communicate it's state to the parent process. This is tricky to do correctly, thankfully, we can use the :class:`PoolProgress <spsim.utils.progress.PoolProgress>` utility for this:

.. literalinclude:: ../../../examples/renderpool_progressbar.py 
   :language: python 

To properly track the progress, we queue render jobs with ``apply_async`` which gives control back to the main process. Then we call ``progress.wait()``, which blocks until all jobs have completed and updates the progress bar accordingly.    

.. important:: After queuing jobs, you *must* wait for them to terminate by calling :meth:`PoolProgress.wait <spsim.utils.progress.PoolProgress.wait>`, otherwise the main process will exit and force-kill all child processes. 
