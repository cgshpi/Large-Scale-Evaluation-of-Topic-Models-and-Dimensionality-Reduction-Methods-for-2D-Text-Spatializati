def get_ignore_words():
    # Set of ignoreWords in C (http://www.c4learn.com/c-programming/c-keywords/)
    ignoreWordsC = {'auto', 'break', 'case', 'char', 'const', 'continue',
                    'default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto',
                    'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof',
                    'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void',
                    'volatile', 'while'}

    # Set of ignoreWords in C++ (https://www.w3schools.in/cplusplus-tutorial/keywords/)
    ignoreWordsCpp = {'h', 'hpp', 'inl', 'alignas', 'alignof', 'and', 'asm',
                      'auto', 'bool', 'break', 'case', 'catch', 'char', 'class', 'const', 'continue',
                      'decltype', 'default', 'delete', 'do', 'double', 'else', 'enum', 'explicit',
                      'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long',
                      'mutable', 'namespace', 'new', 'noexcept', 'not', 'nullptr', 'operator',
                      'or', 'private', 'protected', 'public', 'register', 'return', 'short',
                      'signed', 'sizeof', 'static', 'switch', 'template', 'this', 'throw', 'true',
                      'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
                      'virtual', 'void', 'volatile', 'while', 'xor', 'override', 'final', 'elif',
                      'endif', 'ifdef', 'ifndef', 'define', 'undef', 'include', 'line', 'error',
                      'pragma', 'defined'}

    # Set of ignoreWords in Java (https://www.w3schools.com/java/java_ref_keywords.asp)
    ignoreWordsJava = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case',
                       'catch', 'char', 'class', 'continue', 'const', 'default', 'do', 'double',
                       'else', 'enum', 'exports', 'extends', 'final', 'finally', 'float', 'for',
                       'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface',
                       'long', 'module', 'native', 'new', 'package', 'private', 'protected',
                       'public', 'requires', 'return', 'short', 'static', 'strictfp', 'super',
                       'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try',
                       'var', 'void', 'volatile', 'while'}

    # Set of ignoreWords in JavaScript (https://www.w3schools.com/JS/js_reserved.asp)
    ignoreWordsJavaScript = {'abstract', 'arguments', 'await', 'boolean', 'break',
                             'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 'debugger',
                             'default', 'delete', 'do', 'double', 'else', 'enum', 'eval', 'export',
                             'extends', 'false', 'final', 'finally', 'float', 'for', 'function', 'goto',
                             'if', 'implements', 'import', 'in', 'instanceof', 'int', 'interface', 'let',
                             'long', 'native', 'new', 'null', 'package', 'private', 'protected', 'public',
                             'return', 'short', 'static', 'super', 'switch', 'synchronized', 'this',
                             'throw', 'throws', 'transient', 'true', 'try', 'typeof', 'var', 'void',
                             'volatile', 'while', 'with', 'yield'}

    # Set of ignoreWords in PHP (https://www.php.net/manual/en/reserved.keywords.php)
    ignoreWordsPHP = {'halt', 'compiler', 'abstract', 'and', 'array', 'as',
                      'break', 'callable', 'case', 'catch', 'class', 'clone', 'const', 'continue',
                      'declare', 'default', 'die', 'do', 'echo', 'else', 'elseif', 'empty',
                      'enddeclare', 'endfor', 'endforeach', 'endif', 'endswitch', 'endwhile',
                      'eval', 'exit', 'extends', 'final', 'finally', 'for', 'foreach', 'function',
                      'global', 'goto', 'if', 'implements', 'include', 'once', 'instanceof',
                      'insteadof', 'interface', 'isset', 'list', 'namespace', 'new', 'or', 'print',
                      'private', 'protected', 'public', 'require', 'return', 'static', 'switch',
                      'throw', 'trait', 'try', 'unset', 'use', 'var', 'while', 'xor', 'yield', 'from'}

    # Set of ignoreWords in Python (https://www.w3schools.com/python/python_ref_keywords.asp)
    ignoreWordsPython = {'and', 'as', 'assert', 'break', 'class', 'continue',
                         'def', 'del', 'elif', 'else', 'except', 'False', 'finally', 'for', 'from',
                         'global', 'if', 'import', 'in', 'is', 'lambda', 'None', 'nonlocal', 'not',
                         'or', 'pass', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield'}

    # Set of ignoreWords in Ruby (https://docs.ruby-lang.org/en/2.2.0/keywords_rdoc.html)
    ignoreWordsRuby = {'encoding', 'line', 'file', 'begin', 'end', 'alias', 'and',
                       'begin', 'break', 'case', 'class', 'def', 'defined', 'do', 'else', 'elsif',
                       'end', 'ensure', 'false', 'for', 'if', 'in', 'module', 'next', 'nil', 'not',
                       'or', 'redo', 'rescue', 'retry', 'return', 'self', 'super', 'then', 'true',
                       'undef', 'unless', 'until', 'when', 'while', 'yield'}

    return (ignoreWordsC | ignoreWordsCpp | ignoreWordsJava | ignoreWordsJavaScript | ignoreWordsPHP
            | ignoreWordsPython | ignoreWordsRuby)
