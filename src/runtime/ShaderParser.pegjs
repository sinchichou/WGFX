{
  // Initialize the Intermediate Representation (IR) structure.
  // 初始化 IR 結構
  const shaderInfo = {
    metadata: { use: {}, capability: {} },
    parameters: [],
    textures: [],
    samplers: [],
    passes: [],
    commonCode: ''
  };

  // Tracks the currently active block type (e.g., 'PARAMETER', 'TEXTURE', 'PASS').
  // 追蹤當前活動的區塊類型 (例如：'PARAMETER', 'TEXTURE', 'PASS')。
  let currentBlock = null;
  // Stores data for the current block being parsed.
  // 儲存當前正在解析的區塊資料。
  let currentData = null;
  // Buffer to accumulate code lines within a 'PASS' block.
  // 用於在 'PASS' 區塊內累積程式碼行的緩衝區。
  let passCodeBuffer = [];

  /**
   * Resolves the canonical resource name from a block's parsed data.
   * 從區塊資料解析出規範的資源名稱。
   * Priority: explicit declaration line > //!VAR directive > block header id.
   * 優先順序：宣告行 > //!VAR 指令 > 區塊標頭名稱。
   * @param {object} data - The current block's data object.
   * @returns {string|null} The resolved name, or null if unnamed.
   */
  function resolveName(data) {
    if (data.name) return data.name;
    if (data.var) return data.var;
    const id = (data.id || '').trim();
    return id.length > 0 ? id : null;
  }

  /**
   * Strips surrounding double quotes from a directive value.
   * 移除指令值前後的雙引號。
   * @param {string} value - Raw directive value.
   * @returns {string} Dequoted value.
   */
  function dequote(value) {
    const v = value.trim();
    if (v.length >= 2 && v.startsWith('"') && v.endsWith('"')) {
      return v.slice(1, -1);
    }
    return v;
  }

  /**
   * Commits the parsed data of a block to the main shaderInfo structure.
   * 將解析後的區塊資料提交到主要的 shaderInfo 結構中。
   * @param {string} blockType - The type of the block (e.g., 'PARAMETER', 'COMMON').
   * @param {object} data - The parsed data for the block.
   */
  function commitBlock(blockType, data) {
    if (!blockType || !data || data.isTemplate) {
      currentData = null;
      return;
    }
    switch(blockType) {
      case 'PARAMETER': {
        const name = resolveName(data);
        if (!name) { currentData = null; return; } // Skip unnamed parameter blocks / 略過無名稱的參數區塊
        data.name = name;
        if (!data.type) data.type = 'float';
        delete data.id;
        delete data.lines;
        shaderInfo.parameters.push(data);
        break;
      }
      case 'TEXTURE': {
        const name = resolveName(data);
        if (!name) { currentData = null; return; } // Skip unnamed texture blocks / 略過無名稱的紋理區塊
        data.name = name;
        delete data.id;
        delete data.lines;
        shaderInfo.textures.push(data);
        break;
      }
      case 'SAMPLER': {
        const name = resolveName(data);
        if (!name) { currentData = null; return; } // Skip unnamed sampler blocks / 略過無名稱的取樣器區塊
        data.name = name;
        delete data.id;
        delete data.lines;
        shaderInfo.samplers.push(data);
        break;
      }
      case 'COMMON': shaderInfo.commonCode = data.lines.join('\n'); break;
      case 'PASS': shaderInfo.passes.push(data); break;
      case 'VERSION': shaderInfo.metadata.version = parseInt(data.id,10); break;
      case 'SORT_NAME': shaderInfo.metadata.sortName = dequote(data.id); break;
      case 'USE': data.id.split(',').forEach(f=>shaderInfo.metadata.use[f.trim().toUpperCase()]=true); break;
      case 'CAPABILITY': data.id.split(',').forEach(f=>shaderInfo.metadata.capability[f.trim().toUpperCase()]=true); break;
    }
    currentData = null;
  }

  /**
   * Parses sub-directives within a block (e.g., //! DEFAULT, //! MIN for PARAMETER).
   * 解析區塊內部的子指令 (例如 PARAMETER 的 //! DEFAULT, //! MIN)。
   * @param {object} data - The current block's data object to populate.
   * @param {string} directive - The name of the sub-directive.
   * @param {string} value - The value of the sub-directive.
   */
  function parseSubDirective(data, directive, value) {
    const key = directive.toLowerCase();
    switch(key){
      case 'in':
      case 'out':
        data[key] = value.split(',').map(s=>s.trim()); break;
      case 'block_size':
        data.blockSize = value.split(',').map(Number);
        if(data.blockSize.length===1) data.blockSize.push(data.blockSize[0]); // Handle single value for block_size
        break;
      case 'num_threads':
        data.numThreads = value.split(',').map(Number);
        while(data.numThreads.length<3) data.numThreads.push(1); // Ensure 3 components for num_threads
        break;
      case 'default':
      case 'min':
      case 'max':
      case 'step':
        data[key] = parseFloat(value); break;
      case 'format': {
        // D3D/Magpie 格式名 → WebGPU 格式名（其餘原樣傳遞，交由裝置驗證）
        const FORMAT_MAP = {
          r8g8b8a8_unorm: 'rgba8unorm',
          r8g8b8a8_uint: 'rgba8uint',
          r8g8b8a8_snorm: 'rgba8snorm',
          b8g8r8a8_unorm: 'bgra8unorm',
          r16g16b16a16_float: 'rgba16float',
          r16g16b16a16_unorm: 'rgba16uint',
          r16g16b16a16_snorm: 'rgba16snorm',
          r32g32b32a32_float: 'rgba32float',
          r10g10b10a2_unorm: 'rgb10a2unorm',
        };
        const lower = value.toLowerCase();
        data[key] = FORMAT_MAP[lower] !== undefined ? FORMAT_MAP[lower] : value;
        break;
      }
      case 'label':
      case 'desc':
        data[key] = dequote(value); break;
      case 'type':
        data[key] = value.trim(); break;
      default:
        data[key] = value;
    }
  }
}

start
  = line* {
      // Commit the last block if it was a PASS and had accumulated code.
      // 如果最後一個區塊是 PASS 並且有累積的程式碼，則提交它。
      if (currentBlock === 'PASS' && passCodeBuffer.length > 0) {
        currentData.code = passCodeBuffer.join('\n');
      }
      // Commit any remaining current block data.
      // 提交任何剩餘的當前區塊資料。
      if (currentData) commitBlock(currentBlock, currentData);
      return shaderInfo;
  }

line
  = directive_line /* Matches a directive line (e.g., //! PARAMETER) / 匹配指令行 (例如：//! PARAMETER) */
  / code_line      /* Matches a regular code line / 匹配普通程式碼行 */
  / "\n" { return { type: "empty" }; } /* Matches an empty line / 匹配空行 */

directive_line
  = "//!" _ name:identifier _ value:rest_of_line {
      const directive = name.toUpperCase();
      const val = value.trim();

      // Handle the 'END' directive, which explicitly closes the current block.
      // 處理 'END' 指令，它會明確關閉當前區塊。
      if (directive === 'END') {
          // If a PASS block was active, commit its accumulated code.
          // 如果 PASS 區塊處於活動狀態，則提交其累積的程式碼。
          if (currentBlock === 'PASS' && passCodeBuffer.length > 0) {
              currentData.code = passCodeBuffer.join('\n');
              passCodeBuffer = [];
          }
          // Commit any remaining current block data before ending.
          // 在結束前提交任何剩餘的當前區塊資料。
          if (currentData) {
              commitBlock(currentBlock, currentData);
          }
          currentBlock = null;
          currentData = null;
          return { type: "directive", directive: "END" };
      }

      // Check for block-starting directives.
      // 檢查區塊起始指令。
      if (["PARAMETER","TEXTURE","SAMPLER","COMMON","PASS", "VERSION","SORT_NAME","USE","CAPABILITY"].includes(directive)) {
          // If we were in a PASS block, commit its code first before starting a new block.
          // 如果我們在 PASS 區塊中，則在開始新區塊之前先提交其程式碼。
          if (currentBlock === 'PASS' && passCodeBuffer.length > 0) {
            currentData.code = passCodeBuffer.join('\n');
            passCodeBuffer = [];
          }
          // Commit the data of the previous block.
          // 提交前一個區塊的資料。
          if (currentData) {
            commitBlock(currentBlock, currentData);
          }

          // Start a new block based on the directive.
          // 根據指令開始一個新區塊。
          if (["PARAMETER","TEXTURE","SAMPLER","COMMON","PASS"].includes(directive)) {
            currentBlock = directive;
            currentData = { id: val, lines: [] }; // Initialize currentData for the new block.
            if (directive === 'PASS') currentData.index = parseInt(val,10); // Special handling for PASS index.
          } else { // Global directives (VERSION, SORT_NAME, USE, CAPABILITY) are self-contained and don't start a multi-line block.
            currentBlock = null;
            currentData = null;
            commitBlock(directive, { id: val }); // Commit immediately as they are single-line directives.
          }
      } else { // This is a sub-directive within an active block.
        // If there's no active block, this sub-directive is out of place.
        // 如果沒有活動區塊，則此子指令放置不當。
        if (!currentData) throw new Error(`Directive //! ${directive} cannot be used here / 指令 //! ${directive} 不能在此處使用`);
        parseSubDirective(currentData, directive, val); // Parse the sub-directive.
      }
      return { type: "directive", directive: directive, value: val };
    }

code_line
  = text:[^\n]+ "\n"? {
      const lineStr = text.join("");
      const trimmedLine = lineStr.trim();

      // Special handling for resource declarations in TEXTURE, SAMPLER, PARAMETER blocks.
      // 支援兩種宣告風格：
      //   (a) WGSL 風格:  var Name: texture_2d<f32>; / var Name: sampler;
      //   (b) Magpie/HLSL 風格: Texture2D Name; / SamplerState Name; / float Name; / vec2 Name;
      // 對於 TEXTURE, SAMPLER, PARAMETER 區塊中的資源宣告進行特殊處理。
      if (currentBlock === 'TEXTURE' || currentBlock === 'SAMPLER' || currentBlock === 'PARAMETER') {
        // (a) WGSL-style: var <name> :
        const isComment = trimmedLine.startsWith('//');
        const wgslMatch = isComment ? null : trimmedLine.match(/var\s+([A-Za-z_][A-Za-z0-9_]*)\s*:/);
        // (b) HLSL-style: <TypeKeyword> <name> ;
        const hlslMatch = isComment ? null : trimmedLine.match(/^\s*(Texture2D|SamplerState|float|vec2|vec3|vec4|int|uint|f32|i32|u32)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;/);
        const declMatch = wgslMatch || hlslMatch;
        if (declMatch) {
            currentData.isTemplate = true; // Mark as a template for resource creation.
            const name = wgslMatch ? wgslMatch[1] : hlslMatch[2]; // Extract the resource name.
            const keyword = wgslMatch ? null : hlslMatch[1];      // HLSL type keyword (if any).

            // Derive the parameter type from the HLSL keyword / 由 HLSL 關鍵字推導參數型別
            let paramType = null;
            if (currentBlock === 'PARAMETER' && keyword) {
              paramType = { 'float': 'float', 'f32': 'float', 'vec2': 'vec2', 'vec3': 'vec3', 'vec4': 'vec4', 'int': 'int', 'i32': 'int', 'uint': 'uint', 'u32': 'uint' }[keyword] || 'float';
            }

            const newResource = { ...currentData, name: name, id: name }; // Create a new resource object.
            if (paramType) newResource.type = paramType;
            if (currentBlock === 'TEXTURE' && keyword === 'Texture2D') newResource.declStyle = 'hlsl';
            if (currentBlock === 'SAMPLER' && keyword === 'SamplerState') newResource.declStyle = 'hlsl';
            delete newResource.lines; // These are specific to COMMON blocks.
            delete newResource.isTemplate;

            // Push the new resource to the appropriate shaderInfo array.
            // 將新資源推送到相應的 shaderInfo 陣列中。
            if (currentBlock === 'TEXTURE') {
                shaderInfo.textures.push(newResource);
            } else if (currentBlock === 'SAMPLER') {
                shaderInfo.samplers.push(newResource);
            } else if (currentBlock === 'PARAMETER') {
                shaderInfo.parameters.push(newResource);
            }
            return { type: "code", text: lineStr };
        }
      }

      // Accumulate code lines for COMMON and PASS blocks.
      // 累積 COMMON 和 PASS 區塊的程式碼行。
      if (currentBlock === 'COMMON') {
        currentData.lines.push(lineStr);
      } else if (currentBlock === 'PASS') {
        passCodeBuffer.push(lineStr);
      }
      return { type: "code", text: lineStr };
  }

identifier
  = $([a-zA-Z0-9_]+) /* Matches an identifier (alphanumeric and underscore) / 匹配識別符 (字母數字和底線) */

rest_of_line
  = $([^\n]*) /* Matches the rest of the line until a newline character / 匹配直到換行符為止的行其餘部分 */

_ = [ \t]* /* Matches zero or more spaces or tabs (whitespace) / 匹配零個或多個空格或 tab (空白字元) */
