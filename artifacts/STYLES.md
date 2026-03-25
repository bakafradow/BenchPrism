# Styles Supported by BenchPrism

Here are all 41 supported coding styles mentioned in the paper, in consistent with `settings/style_options.yaml`.

## Formatting

### Indention[^1][^2]

* **Indention Unit (`indention_unit`)**: Determines the basic indentation character and size. Options are `TAB`, `TWO_SPACES`, or `FOUR_SPACES`.

### Space[^1][^2]

* **Operator Spacing (`operator_spacing`)**: Toggles spaces around operators (e.g., `a+b` vs `a + b`).
* **Call Spacing (`call_spacing`)**: Toggles spaces in method/function calls.
* **Comma Spacing (`comma_spacing`)**: Toggles spaces after commas.
* **Semicolon Spacing (`semicolon_spacing`)**: Toggles spaces after semicolons.

### Newline[^1][^2]

* **Member Padding (`member_padding`)**: Blank lines around class members.
* **Block Padding (`block_padding`)**: Blank lines around code blocks.
* **Declaration Padding (`declaration_padding`)**: Blank lines around variable/method declarations.
* **Line Wrapping (`long_line_wrapping`)**: A boolean that determines if long lines of code exceeding the maximum length should be wrapped to the next line.
* **Break Before Brace (`break_before_brace`)**: A boolean controlling brace placement. If true, the opening brace `{` is placed on a new line (Allman style). If false, it stays on the same line.

## Structural

### Optional Braces[^1][^2][^5]

* **Omit Braces (`omit_braces`)**: Controls the `useBrace` property. If true, braces `{}` are omitted for single-statement bodies (e.g., `if (cond) stmt;`).

### Modifier Order[^1]

* **Modifier Order (`order`)**: Defines the sequence of Java modifiers (e.g., `public static` vs `static public`). Options are `STANDARD` or `REVERSED`.

### Declaration Layout[^1][^2][^3][^4]

* **Declaration Merging (`merge_declarations`)**: A boolean controlling `mergeVar`. If true, adjacent variables of the same type are merged (e.g., `int a; int b;` becomes `int a, b;`).

### Increment/Decrement[^3]

* **Increment Statement (`1` in StructureStyler)**: Chooses between 5 increment styles: `$E++`, `++$E`, `$E+=1`, `$E=$E1+1`, or `$E=1+$E1`.
* **Decrement Statement (`2` in StructureStyler)**: Chooses between 4 decrement styles: `$E--`, `--$E`, `$E-=1`, or `$E=$E1-1`.

### Assignment[^3]

* **Assignment Statement (`3` in StructureStyler)**: Chooses between compound (`$E += $E2`) and regular assignment (`$E = $E1 + $E2`).

### Loops[^3][^4][^5]

* **Empty Loop (`6` in StructureStyler)**: Empty `for(;;)` vs. `while(true)`.
* **Loop Condition Only (`7` in StructureStyler)**: Condition-only `for(;$E;)` vs. `while($E)`.
* **Complex Loops (`8` to `12` in StructureStyler)**: Controls preferences for complex loops, converting between `while` loops and `for` loops with varying combinations of initializers, declarations, conditions, and update statements. (Lengths vary from 2 to 3).

### Operators[^4]

* **Operator Preference 1 (`16` in StructureStyler)**: Less-than (`$E < $E1`) vs. Greater-than (`$E1 > $E`).
* **Operator Preference 2 (`17` in StructureStyler)**: Less-than-equal (`$E <= $E1`) vs. Greater-than-equal (`$E1 >= $E`).
* **Operator Preference 3 (`18` in StructureStyler)**: Explicit comparison (`$E == false`) vs. negation operator (`!$E`).

### Literals[^4]

* **Literal Position (`19` in StructureStyler)**: "Yoda" conditions (`$LITERAL == $E`) vs. standard (`$E == $LITERAL`).

### Arrays[^1]

* **Array Declaration (`20` and `21` in StructureStyler)**: Bracket placement on the type (`T[] I`) vs. the identifier (`T I[]`), for both declarations and initializations.

## Semantic

### Conditional Statement Order[^4]

* **Short Body Comes First (`short_body_comes_first`)**: A boolean determining if the shorter block of code in an if-else structure should be placed in the `if` body rather than the `else` body.

### Naming[^1][^2][^3]

* **Case Format (`case_format`)**: Controls identifier casing. Options are `UPPER_UNDERSCORE`, `LOWER_UNDERSCORE`, `UPPER_CAMEL`, and `LOWER_CAMEL`.
* **Brief Naming (`brief`)**: A boolean controlling the `maxLength` property to enforce shorter variable names.

### Compound If Statement[^3][^4][^5]

* **Nesting If (`4` in StructureStyler)**: Merged logic (`if ($E && $E1)`) vs. nested ifs (`if ($E) { if ($E1) }`).
* **Cascading If (`5` in StructureStyler)**: Merged logic (`if ($E || $E1)`) vs. sequential ifs (`if ($E) S; if ($E1) S;`).

### Conditional Return[^2]

* **Conditional Return (`13` in StructureStyler)**: Ternary return (`return $E ? $E1 : $E2;`) vs. if-else return (`if ($E) return $E1; else return $E2;`).

### Conditional Assignment[^4]

* **Declare Then Check and Assign (`14` in StructureStyler)**: Chooses between 4 styles for declaring a variable and assigning it conditionally (using ternary operators or if-else blocks).
* **Check Then Assign (`15` in StructureStyler)**: Similar to 14, but for existing variables (re-assignment).

### Loop Guard Clause & Redundant Body[^6]

* **Loop Guard Clause (`22` and `23` in StructureStyler)**: Chooses how `continue` statements are used inside `if` blocks versus utilizing `else` blocks.
* **Redundant Body (`24` and `25` in StructureStyler)**: Simplifies unconditional code execution after `if` or `if-else` blocks (e.g., removing redundant `else` structures).

---

[^1]: Google. n.d. Google Java style guide. Retrieved September 28, 2025 from https://google.github.io/styleguide/javaguide.html

[^2]: Oracle. n.d. Code Conventions for the Java Programming Language. Retrieved September 28, 2025 from https://www.oracle.com/java/technologies/javase/codeconventions-contents.html

[^3]: Zhen Li, Guenevere Chen, Chen Chen, Yayi Zou, and Shouhuai Xu. 2022. Ropgen: Towards robust code authorship attribution via automatic coding style transformation. In Proceedings of the 44th International Conference on Software Engineering, 1906–1918.

[^4]: Qianjun Liu, Shouling Ji, Changchang Liu, and Chunming Wu. 2021. A practical black-box attack on source code authorship identification classifiers. IEEE Transactions on Information Forensics and Security 16 (2021), 3620–3633.

[^5]: Erwin Quiring, Alwin Maier, and Konrad Rieck. 2019. Misleading authorship attribution of source code using adversarial learning. In 28th USENIX Security Symposium (USENIX Security 19), 479–496.

[^6]: Binger Chen and Ziawasch Abedjan. 2023. DuetCS: Code Style Transfer through Generation and Retrieval. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE). IEEE, 2362–2373.
