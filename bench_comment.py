import os
import subprocess

pr_nr = os.environ.get("PR_NR")
diff_result = subprocess.run(
    ["asv", "--config=benchmark/asv.conf.json", "compare", "upstream/main", "HEAD"],
    capture_output=True,
)
comment = f"""\
[Benchmark results](https://s-weigand.github.io/pyglotaran-benchmarks/prs/pr-{pr_nr})
<details>
<summary>
Benchmark diff
</summary>

```
{diff_result.stdout.decode(encoding="cp850")}
```
</details>
"""

print(f"::set-output name=comment::{comment}")
