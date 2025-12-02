{
  // 初始化 IR 結構
  const shaderInfo = {
    metadata: { use: {}, capability: {} },
    parameters: [],
    textures: [],
    samplers: [],
    passes: [],
    commonCode: ''
  };

  let currentBlock = null;
  let currentData = null;
  let passCodeBuffer = [];
}

start
  = line* {
      // 提交最後一個區塊
      if (currentBlock === 'PASS' && passCodeBuffer.length > 0) {
        currentData.code = passCodeBuffer.join('\n');
      }
      if (currentData) commitBlock(currentBlock, currentData);
      return shaderInfo;
  }

line
  = directive_line
  / code_line
  / "\n" { return { type: "empty" }; }

directive_line
  = "//!" _ name:identifier _ value:rest_of_line {
      const directive = name.toUpperCase();
      const val = value.trim();

      if (["PARAMETER","TEXTURE","SAMPLER","COMMON","PASS"].includes(directive)) {
        if (currentData) commitBlock(currentBlock, currentData);
        currentBlock = directive;
        currentData = { id: val, lines: [] };
        if (directive === 'PASS') currentData.index = parseInt(val,10);
      } else if (["VERSION","SORT_NAME","USE","CAPABILITY"].includes(directive)) {
        if (currentData) commitBlock(currentBlock, currentData);
        currentBlock = null;
        currentData = null;
        commitBlock(directive, { id: val });
      } else {
        if (!currentData) throw new Error(`指令 //! ${directive} 不能在此處使用`);
        parseSubDirective(currentData, directive, val);
      }
      return { type: "directive", directive, value: val };
    }

code_line
  = text:[^\n]+ "\n"? {  // 改成 +，至少匹配一個字元
      const lineStr = text.join("");
      if (currentBlock === 'COMMON') {
        currentData.lines.push(lineStr);
      } else if (currentBlock === 'PASS') {
        passCodeBuffer.push(lineStr);
      }
      return { type: "code", text: lineStr };
  }

identifier
  = [a-zA-Z0-9_]+

rest_of_line
  = [^\n]*

_ = [ \t]*

/////////////////////////
// Helper functions
/////////////////////////

{
function commitBlock(blockType, data) {
  if (!blockType || !data) return;
  switch(blockType) {
    case 'PARAMETER': shaderInfo.parameters.push(data); break;
    case 'COMMON': shaderInfo.commonCode = data.lines.join('\n'); break;
    case 'TEXTURE': shaderInfo.textures.push(data); break;
    case 'SAMPLER': shaderInfo.samplers.push(data); break;
    case 'PASS': shaderInfo.passes.push(data); break;
    case 'VERSION': shaderInfo.metadata.version = parseInt(data.id,10); break;
    case 'SORT_NAME': shaderInfo.metadata.sortName = data.id; break;
    case 'USE': data.id.split(',').forEach(f=>shaderInfo.metadata.use[f.trim().toUpperCase()]=true); break;
    case 'CAPABILITY': data.id.split(',').forEach(f=>shaderInfo.metadata.capability[f.trim().toUpperCase()]=true); break;
  }
  currentData = null;
}

function parseSubDirective(data, directive, value) {
  const key = directive.toLowerCase();
  switch(key){
    case 'in':
    case 'out':
      data[key] = value.split(',').map(s=>s.trim()); break;
    case 'block_size':
      data.blockSize = value.split(',').map(Number);
      if(data.blockSize.length===1) data.blockSize.push(data.blockSize[0]);
      break;
    case 'num_threads':
      data.numThreads = value.split(',').map(Number);
      while(data.numThreads.length<3) data.numThreads.push(1);
      break;
    case 'default':
    case 'min':
    case 'max':
    case 'step':
      data[key] = parseFloat(value); break;
    case 'format':
      data[key] = value.toLowerCase()==='r16g16b16a16_float' ? 'rgba16float' : value;
      break;
    default:
      data[key] = value;
  }
}
}
