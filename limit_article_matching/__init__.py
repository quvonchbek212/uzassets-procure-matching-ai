"""Product limit-article matching service.

Assigns a budget limit article to a product, so the backend can store the
mapping and reuse it for every purchase-request line that carries that
product. Reads UzAssets with the read-only `read-user` credential and never
writes: storing mappings, the user's selection and the line update all stay
with the backend.

Import the router as `from limit_article_matching.router import router`.
Nothing is re-exported here on purpose — binding the name `router` in the
package namespace would shadow the `limit_article_matching.router` submodule.

See README.md in this package for the contract.
"""

from __future__ import annotations

__all__: list[str] = []
