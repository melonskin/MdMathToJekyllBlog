# MdMathToJekyllBlog

My blog is powered by Jekyll. Latex math is supported using Mathjax. For some reason, the math in raw markdown file cannot be rendered properly by this combination. For example as inline math `\(\alpha\)`,Markdown will try to render `\ ... \` first, which leaves `(\alpha)`. The remaining content cannot be recognized by Mathjax. So the script converts `\(\alpha\)` to `\\(\alpha\\)` in order to solve the problem. Some other cases are taken care of as well.
