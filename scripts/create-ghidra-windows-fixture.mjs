import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
for (const architecture of ["x86_64", "x86"]) {
  const x86 = architecture === "x86";
  const output = resolve(
    root,
    "build",
    "fixtures",
    x86 ? "rea-ghidra-windows-x86.exe" : "rea-ghidra-windows.exe",
  );
  const bytes = Buffer.alloc(1536);
  const peOffset = 0x80;
  const optionalHeader = peOffset + 24;
  const optionalHeaderSize = x86 ? 0xe0 : 0xf0;
  const sectionTable = optionalHeader + optionalHeaderSize;

  bytes.write("MZ", 0, "ascii");
  bytes.writeUInt32LE(peOffset, 0x3c);
  bytes.write("PE\0\0", peOffset, "binary");
  bytes.writeUInt16LE(x86 ? 0x14c : 0x8664, peOffset + 4);
  bytes.writeUInt16LE(2, peOffset + 6);
  bytes.writeUInt16LE(optionalHeaderSize, peOffset + 20);
  bytes.writeUInt16LE(x86 ? 0x102 : 0x22, peOffset + 22);

  bytes.writeUInt16LE(x86 ? 0x10b : 0x20b, optionalHeader);
  bytes.writeUInt32LE(0x200, optionalHeader + 4);
  bytes.writeUInt32LE(0x200, optionalHeader + 8);
  bytes.writeUInt32LE(0x1000, optionalHeader + 16);
  bytes.writeUInt32LE(0x1000, optionalHeader + 20);
  if (x86) bytes.writeUInt32LE(0x400000, optionalHeader + 28);
  else bytes.writeBigUInt64LE(0x1_4000_0000n, optionalHeader + 24);
  bytes.writeUInt32LE(0x1000, optionalHeader + 32);
  bytes.writeUInt32LE(0x200, optionalHeader + 36);
  bytes.writeUInt16LE(6, optionalHeader + 40);
  bytes.writeUInt16LE(6, optionalHeader + 48);
  bytes.writeUInt32LE(0x3000, optionalHeader + 56);
  bytes.writeUInt32LE(0x200, optionalHeader + 60);
  bytes.writeUInt16LE(3, optionalHeader + 68);
  bytes.writeUInt16LE(x86 ? 0x8140 : 0x8160, optionalHeader + 70);
  if (x86) {
    bytes.writeUInt32LE(0x10_0000, optionalHeader + 72);
    bytes.writeUInt32LE(0x1000, optionalHeader + 76);
    bytes.writeUInt32LE(0x10_0000, optionalHeader + 80);
    bytes.writeUInt32LE(0x1000, optionalHeader + 84);
    bytes.writeUInt32LE(16, optionalHeader + 92);
  } else {
    bytes.writeBigUInt64LE(0x10_0000n, optionalHeader + 72);
    bytes.writeBigUInt64LE(0x1000n, optionalHeader + 80);
    bytes.writeBigUInt64LE(0x10_0000n, optionalHeader + 88);
    bytes.writeBigUInt64LE(0x1000n, optionalHeader + 96);
    bytes.writeUInt32LE(16, optionalHeader + 108);
  }

  bytes.write(".text", sectionTable, "ascii");
  bytes.writeUInt32LE(0x50, sectionTable + 8);
  bytes.writeUInt32LE(0x1000, sectionTable + 12);
  bytes.writeUInt32LE(0x200, sectionTable + 16);
  bytes.writeUInt32LE(0x200, sectionTable + 20);
  bytes.writeUInt32LE(0x6000_0020, sectionTable + 36);

  const dataSection = sectionTable + 40;
  bytes.write(".data", dataSection, "ascii");
  bytes.writeUInt32LE(0x20, dataSection + 8);
  bytes.writeUInt32LE(0x2000, dataSection + 12);
  bytes.writeUInt32LE(0x200, dataSection + 16);
  bytes.writeUInt32LE(0x400, dataSection + 20);
  bytes.writeUInt32LE(0xc000_0040, dataSection + 36);
  bytes.writeUInt32LE(0x1122_3344, 0x400);

  // Entry reaches both functions; the original callee still returns 42.
  // These source-owned bytes are inspected statically, never executed.
  bytes.set(
    [0xe8, 0x0b, 0, 0, 0, 0xe8, 0x16, 0, 0, 0, 0x31, 0xc0, 0xc3],
    0x200,
  );
  bytes.set([0xb8, 0x2a, 0, 0, 0, 0xc3], 0x210);
  const referenceInstructions = x86
    ? [
        "ba00204000", // MOV EDX, address: DATA only.
        "a100204000", // MOV EAX, [address]: READ.
        "a304204000", // MOV [address], EAX: WRITE.
        "ba10204000", // Indexed base, unknown ECX.
        "85db", // TEST EBX, EBX: two candidate bases.
        "7405", // JZ over the alternate base.
        "ba18204000", // MOV EDX, alternate base.
        "c6040a01", // MOV byte [EDX + ECX], 1.
        "ffd7", // CALL EDI: unknown target.
        "c3",
      ]
    : [
        "488d15d90f0000", // LEA RDX, [RIP + displacement]: DATA.
        "8b05d30f0000", // MOV EAX, [RIP + displacement]: READ.
        "8905d10f0000", // MOV [RIP + displacement], EAX: WRITE.
        "488d15d60f0000", // Indexed base, unknown RCX.
        "85db", // TEST EBX, EBX: two candidate bases.
        "7407", // JZ over the alternate base.
        "488d15d30f0000", // LEA RDX, alternate base.
        "c6040a01", // MOV byte [RDX + RCX], 1.
        "41ffd0", // CALL R8: unknown target.
        "c3",
      ];
  bytes.set(Buffer.from(referenceInstructions.join(""), "hex"), 0x220);

  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, bytes);
  if (!x86) process.stdout.write(`${output}\n`);
}
