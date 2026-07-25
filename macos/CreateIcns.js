const fs = require("fs");
const path = require("path");

const iconsetPath = process.argv[2];
const outputPath = process.argv[3];

const sources = [
  ["ic10", "icon_512x512@2x.png"],
  ["ic09", "icon_512x512.png"],
  ["ic08", "icon_256x256.png"],
  ["ic07", "icon_128x128.png"],
  ["icp5", "icon_32x32.png"],
  ["icp4", "icon_16x16.png"]
];

const chunks = sources.map(([type, filename]) => {
  const data = fs.readFileSync(path.join(iconsetPath, filename));
  const chunk = Buffer.alloc(8 + data.length);
  chunk.write(type, 0, 4, "ascii");
  chunk.writeUInt32BE(chunk.length, 4);
  data.copy(chunk, 8);
  return chunk;
});

const totalLength = 8 + chunks.reduce((sum, chunk) => sum + chunk.length, 0);
const header = Buffer.alloc(8);
header.write("icns", 0, 4, "ascii");
header.writeUInt32BE(totalLength, 4);
fs.writeFileSync(outputPath, Buffer.concat([header, ...chunks], totalLength));
