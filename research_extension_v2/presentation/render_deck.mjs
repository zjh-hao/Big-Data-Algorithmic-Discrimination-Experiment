import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));
const input=process.argv[2]??path.join(here,'output/Fairness_Research_V2.pptx');
const presentation=await PresentationFile.importPptx(await FileBlob.load(input));
await fs.mkdir(path.join(here,'previews'),{recursive:true});
for(let i=0;i<presentation.slides.items.length;i++){
 const blob=await presentation.export({slide:presentation.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(here,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log(`Rendered ${presentation.slides.items.length} final-file slides`);
