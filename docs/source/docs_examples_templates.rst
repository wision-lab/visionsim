===================
Code Block Examples
===================

Basic Code Block
================

.. code-block:: python

    def foo(bars):
        for bar in bars:
            print("Hello World!")

    foobar = foo(bar)

Code Block with Multiple Languages
==================================

.. warning:: 
    Sphinx documentation says to use the `sphinxcontrib-osexample` to create a code block with multiple languages. The library has a ton of bugs and doesn't work.

Code Block from an Example File
===============================

Entire File
------------
.. literalinclude:: example.py
  :language: python
  :caption: example.py

Specific Lines from File
-------------------------
.. literalinclude:: example.py
  :language: python
  :caption: foo function from example.py
  :lines: 5-7