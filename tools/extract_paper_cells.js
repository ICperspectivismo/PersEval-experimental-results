// Optional macOS maintainer utility, never used by tests/package/site builds.
// osascript -l JavaScript tools/extract_paper_cells.js PAPER.pdf OUTPUT_DIRECTORY
ObjC.import('Foundation');
ObjC.import('PDFKit');
ObjC.import('AppKit');
function run(argv) {
  if (argv.length !== 2) throw new Error('Provide a local paper PDF and an existing output directory.');
  const pdf = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  if (!pdf || pdf.pageCount !== 26) throw new Error('Expected the 26-page local PersEval paper.');
  for (const number of [7, 8, 19, 20, 21, 22]) {
    const page = pdf.pageAtIndex(number - 1), text = ObjC.unwrap(page.string);
    const words = [], pattern = /\.\d{3}|\d+(?:\.\d+)?|[A-Za-z][A-Za-z-]*/g;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const box = page.selectionForRange($.NSMakeRange(match.index, match[0].length)).boundsForPage(page);
      words.push({text: match[0], x: box.origin.x, y: box.origin.y, w: box.size.width, h: box.size.height});
    }
    $.NSString.stringWithString(JSON.stringify(words)).writeToFileAtomicallyEncodingError(
      argv[1] + '/perseval-paper-words-' + number + '.json', true, $.NSUTF8StringEncoding, null);
    const image = page.thumbnailOfSizeForBox($.NSMakeSize(1800, 2400), 0);
    $.NSBitmapImageRep.imageRepWithData(image.TIFFRepresentation).representationUsingTypeProperties(
      4, $.NSDictionary.dictionary).writeToFileAtomically(argv[1] + '/perseval-paper-page-' + number + '.png', true);
  }
  return 'Extracted numeric-cell coordinates and rendered review pages; no network access.';
}
