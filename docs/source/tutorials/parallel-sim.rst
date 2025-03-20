Parallel Simulation
===================

Rendering an animation with a single blender instance can be done like so:

.. literalinclude:: ../../../examples/render_animation.py 
   :language: python   

Making it use multiple instances is as easy as using :class:`BlenderClients <spsim.simulate.blender.BlenderClients>` instead of :class:`BlenderClient <spsim.simulate.blender.BlenderClient>` (notice the ``s``)! Here's all that needs to change: 

.. program-output:: git diff --word-diff=color --word-diff-regex=. --no-index --color ../../examples/render_animation.py ../../examples/render_animation_parallel.py | sed -z 's/.*@@.*@@....//g'
   :language: ansi-python-console
   :shell:

.. warning:: In practice, the number of rendering jobs will be limited by the user's system resources, most likely GPU VRAM. Start small, and increase accordingly.
