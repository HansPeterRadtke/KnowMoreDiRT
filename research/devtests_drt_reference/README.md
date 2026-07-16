# Independent DRT reference implementation and acceptance evidence

This directory preserves selected files from the independent `devtests/DRT_tests` lineage as recovery evidence and porting material. It is not imported by the production package and is not a substitute for the KnowMoreDiRT implementation.

The `pure_raw` snapshot comes from commit `46f6845556ee86efad56ab0a3aa7ba58b04e916f`, the commit identified in the recovery record as the pure-raw DRT model-query scoring implementation. Its corresponding final report was added in commit `5da8b43df75cc6dfee474dcd374083cd817f548f`.

The `public_raw_folder` snapshot comes from commit `f38af1b0d9c191139ed445e172d9ba5210847b1d`, the documented public raw-folder validation lineage.

The recovery target remains a database-backed DRT system in KnowMoreDiRT. These files provide independent contracts, black-box checks, no-overfit audits, and benchmark harness behavior that can be ported into native tests without introducing gold-answer access or replacing DRT with model-owned execution.
