'use strict';

// OCR digit/letter confusions in Hebrew context: maps misread characters → correct Hebrew letters.
// Keys are ASCII characters that OCR commonly substitutes for visually similar Hebrew letters.
const _OCR_HEBREW_FIXES = {
  '0': 'ס',  // ס and 0 are visually similar
  '1': 'י',  // י and 1 are visually similar
  '5': 'ה',  // ה and 5 in some OCR fonts
  '6': 'ו',  // ו and 6 are visually similar
  'l': 'ל',  // ל and lowercase l
  'I': 'י',  // י and uppercase I
  'O': 'ס',  // ס and uppercase O
};

// Matches a sequence that contains at least one Hebrew letter and may include OCR-noise chars.
// Used to limit substitutions to Hebrew-context tokens only.
const _HEBREW_TOKEN_RE = /[\u05D0-\u05EAlIO0-9]+/g;

/**
 * Cleans Hebrew text returned from OCR/Gemini.
 * Applies NFC normalization, removes noise characters, fixes apostrophes,
 * and collapses whitespace.
 *
 * @param {string} text
 * @param {object} [options]
 * @param {boolean} [options.fixOcrDigits=false] - When true, replaces digits/ASCII letters
 *   that are commonly misread by OCR inside Hebrew tokens (e.g. 0→ס, 1→י, 6→ו).
 *   Only applied within tokens that already contain at least one Hebrew letter,
 *   so standalone numbers (dates, plates) are not affected.
 * @returns {string}
 */
function cleanHebrew(text, options = {}) {
  if (!text || typeof text !== 'string') return '';

  // Unicode NFC normalization
  let result = text.normalize('NFC');

  // Fix apostrophes (curly/backtick variants → straight)
  result = result.replace(/[\u2018\u2019\u201A\u201B\u0060]/g, "'");

  // Remove noise: keep Hebrew letters (U+05D0–U+05EA), Hebrew punctuation (U+05F0–U+05F4, U+FB1D–U+FB4E),
  // ASCII letters/digits, common punctuation, spaces and newlines
  result = result.replace(/[^\u05D0-\u05EA\u05F0-\u05F4\uFB1D-\uFB4Ea-zA-Z0-9\s.,\-'"/\\():;?!@#%+*=]/g, '');

  // Fix OCR digit/letter confusions — only inside tokens that contain Hebrew letters
  if (options.fixOcrDigits) {
    result = result.replace(_HEBREW_TOKEN_RE, token => {
      const hasHebrew = /[\u05D0-\u05EA]/.test(token);
      if (!hasHebrew) return token; // pure number — leave untouched
      return token.replace(/[0156lIO]/g, ch => _OCR_HEBREW_FIXES[ch] ?? ch);
    });
  }

  // Collapse multiple spaces/tabs into a single space
  result = result.replace(/[ \t]+/g, ' ');

  // Collapse multiple newlines into a single newline
  result = result.replace(/\n{2,}/g, '\n');

  return result.trim();
}

/**
 * Normalizes any common date format to DD/MM/YYYY.
 * Accepts separators: / . -
 * Two-digit years: <=40 → 20XX, >40 → 19XX.
 * Returns null if the date is invalid or not recognized.
 * @param {string} str
 * @returns {string|null}
 */
function normalizeDate(str) {
  if (!str || typeof str !== 'string') return null;

  // Match DD[sep]MM[sep]YY or DD[sep]MM[sep]YYYY
  const match = str.match(/(\d{1,2})[.\-\/](\d{1,2})[.\-\/](\d{2,4})/);
  if (!match) return null;

  let day   = parseInt(match[1], 10);
  let month = parseInt(match[2], 10);
  let year  = parseInt(match[3], 10);

  // Expand two-digit year
  if (year < 100) {
    year = year <= 40 ? 2000 + year : 1900 + year;
  }

  // Validate ranges
  if (day < 1 || day > 31)     return null;
  if (month < 1 || month > 12) return null;
  if (year < 2000 || year > 2040) return null;

  const dd = String(day).padStart(2, '0');
  const mm = String(month).padStart(2, '0');
  return `${dd}/${mm}/${year}`;
}

/**
 * Normalizes a vehicle license plate.
 * Civilian: XXX-XX-XXX (3-2-3 digits, with hyphens).
 * Military: exactly 6 digits (no hyphens).
 * Returns null if the string does not match either format.
 * @param {string} str
 * @returns {string|null}
 */
function normalizePlate(str) {
  if (!str || typeof str !== 'string') return null;

  // Strip spaces and common noise (dots, underscores)
  const clean = str.replace(/[\s._]/g, '');

  // Try civilian format: 3-2-3 (may already have hyphens or not)
  const civilianMatch = clean.match(/^(\d{3})-?(\d{2})-?(\d{3})$/);
  if (civilianMatch) {
    return `${civilianMatch[1]}-${civilianMatch[2]}-${civilianMatch[3]}`;
  }

  // Try military format: exactly 6 digits (no separators)
  const militaryMatch = clean.match(/^\d{6}$/);
  if (militaryMatch) {
    return clean;
  }

  return null;
}

window.DataProcessor = { cleanHebrew, normalizeDate, normalizePlate };
