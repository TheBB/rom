============
Case objects
============

A case object contains all information necessary to solve instances of
a problem. In particular, a case object must know about

* the reference geometry and its discretization,
* the parameters involved and their legal minimum and maximum values,
* the bases used for the different fields,
* how those bases are transformed to the physical geometry (e.g. the
  Piola transform)
* which parts of the boundary are subjected to Dirichlet conditions,
* the lifting function(s)
* any linear, bilinear or trilinear forms needed to assemble and solve
  problem instances

The first step to successfully developing a Reduced Order Model is to
create a case object representing your problem.

.. autoclass:: aroma.cases.bases.Case
   :members:
   :special-members: __init__
