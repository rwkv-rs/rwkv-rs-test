import { NextResponse } from "next/server";
import { loadTask5Dataset } from "@/lib/files";

export const dynamic = "force-dynamic";

export async function GET() {
  const dataset = await loadTask5Dataset();
  return NextResponse.json(dataset);
}
