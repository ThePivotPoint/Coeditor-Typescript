const { ESLint } = require("eslint");
const fs = require("fs");
const path = require("path");

async function analyzeFile(filePath) {
  try {
    const eslint = new ESLint({
      baseConfig: {
        parser: "@typescript-eslint/parser",
        parserOptions: { 
          ecmaVersion: 2020, 
          sourceType: "module",
          project: null // 不使用 tsconfig.json
        }
      },
      useEslintrc: false
    });

    const results = await eslint.lintFiles([filePath]);
    const sourceCode = fs.readFileSync(filePath, "utf-8");
    const lines = sourceCode.split("\n");

    // 简单的引用分析：提取每行的标识符
    const line2usages = {};
    
    for (let lineNum = 1; lineNum <= lines.length; lineNum++) {
      const line = lines[lineNum - 1];
      const identifiers = extractIdentifiersFromLine(line);
      if (identifiers.length > 0) {
        line2usages[lineNum] = identifiers;
      }
    }

    // 输出到 JSON 文件
    const outputPath = path.join(path.dirname(filePath), "ts_usage_output.json");
    fs.writeFileSync(outputPath, JSON.stringify(line2usages, null, 2));
    
    console.log(`Analysis completed for ${filePath}`);
    console.log(`Output saved to ${outputPath}`);
    
  } catch (error) {
    console.error(`Error analyzing ${filePath}:`, error.message);
    // 如果分析失败，创建空的输出文件
    const outputPath = path.join(path.dirname(filePath), "ts_usage_output.json");
    fs.writeFileSync(outputPath, JSON.stringify({}, null, 2));
  }
}

function extractIdentifiersFromLine(line) {
  // 简单的标识符提取：匹配变量名、函数名、类名等
  const identifierRegex = /\b[a-zA-Z_$][a-zA-Z0-9_$]*\b/g;
  const matches = line.match(identifierRegex) || [];
  
  // 过滤掉关键字和常见的内置对象
  const keywords = [
    'const', 'let', 'var', 'function', 'class', 'interface', 'type', 'import', 'export',
    'if', 'else', 'for', 'while', 'return', 'new', 'this', 'super', 'async', 'await',
    'true', 'false', 'null', 'undefined', 'NaN', 'Infinity'
  ];
  
  return matches.filter(id => !keywords.includes(id));
}

// 从命令行参数获取文件路径
const filePath = process.argv[2];
if (!filePath) {
  console.error("Usage: node analyze_ts_usage.js <typescript_file_path>");
  process.exit(1);
}

analyzeFile(filePath); 