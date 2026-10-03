# Tanzil Quran Survey & Word Stream Line Simulation Report

## 1. Line Count & Basmalah Handling
- **Line Count:** All 5 variants contain exactly **6,236 non-comment Ayah lines**.
- **Comment Lines:** 28 header/footer comment lines (`#`) carrying Tanzil copyright.
- **Basmalah Placement:**
  - Surah 1 (Al-Fatiha): Basmalah is an independent Ayah (Ayah 1).
  - Surahs 2-114 (except Surah 9): Basmalah is prepended to the start of Ayah 1.
  - Surah 9 (At-Tawbah): Does not have a Basmalah.

## 2. Ayah Length Distribution

| Variant | Words (med / p95 / max) | Chars (med / p95 / max) | Ayahs >200 Chars |
| :--- | :--- | :--- | :--- |
| uthmani | 10 / 30 / 128 | 92 / 265 / 1173 | 782 (max at 2:282) |
| uthmani-min | 10 / 30 / 128 | 79 / 229 / 1007 | 504 (max at 2:282) |
| simple | 10 / 30 / 129 | 90 / 259 / 1142 | 732 (max at 2:282) |
| simple-clean | 10 / 30 / 129 | 53 / 154 / 679 | 124 (max at 2:282) |
| simple-min | 10 / 30 / 129 | 78 / 228 / 1003 | 495 (max at 2:282) |

## 3. Codepoint Inventory by Variant

### uthmani (62 distinct codepoints)

| Codepoint | Char | Count | Unicode Name |
| :--- | :---: | :--- | :--- |
| `U+0020` | `[SPACE]` | 71645 | SPACE |
| `U+0621` | `ء` | 2782 | ARABIC LETTER HAMZA |
| `U+0623` | `أ` | 8900 | ARABIC LETTER ALEF WITH HAMZA ABOVE |
| `U+0624` | `ؤ` | 706 | ARABIC LETTER WAW WITH HAMZA ABOVE |
| `U+0625` | `إ` | 5088 | ARABIC LETTER ALEF WITH HAMZA BELOW |
| `U+0626` | `ئ` | 921 | ARABIC LETTER YEH WITH HAMZA ABOVE |
| `U+0627` | `ا` | 25184 | ARABIC LETTER ALEF |
| `U+0628` | `ب` | 11603 | ARABIC LETTER BEH |
| `U+0629` | `ة` | 2344 | ARABIC LETTER TEH MARBUTA |
| `U+062A` | `ت` | 10520 | ARABIC LETTER TEH |
| `U+062B` | `ث` | 1414 | ARABIC LETTER THEH |
| `U+062C` | `ج` | 3317 | ARABIC LETTER JEEM |
| `U+062D` | `ح` | 4364 | ARABIC LETTER HAH |
| `U+062E` | `خ` | 2497 | ARABIC LETTER KHAH |
| `U+062F` | `د` | 5991 | ARABIC LETTER DAL |
| `U+0630` | `ذ` | 4932 | ARABIC LETTER THAL |
| `U+0631` | `ر` | 12627 | ARABIC LETTER REH |
| `U+0632` | `ز` | 1599 | ARABIC LETTER ZAIN |
| `U+0633` | `س` | 6122 | ARABIC LETTER SEEN |
| `U+0634` | `ش` | 2124 | ARABIC LETTER SHEEN |
| `U+0635` | `ص` | 2074 | ARABIC LETTER SAD |
| `U+0636` | `ض` | 1686 | ARABIC LETTER DAD |
| `U+0637` | `ط` | 1273 | ARABIC LETTER TAH |
| `U+0638` | `ظ` | 853 | ARABIC LETTER ZAH |
| `U+0639` | `ع` | 9405 | ARABIC LETTER AIN |
| `U+063A` | `غ` | 1221 | ARABIC LETTER GHAIN |
| `U+0640` | `ـ` | 812 | ARABIC TATWEEL |
| `U+0641` | `ف` | 8747 | ARABIC LETTER FEH |
| `U+0642` | `ق` | 7034 | ARABIC LETTER QAF |
| `U+0643` | `ك` | 10497 | ARABIC LETTER KAF |
| `U+0644` | `ل` | 38550 | ARABIC LETTER LAM |
| `U+0645` | `م` | 27071 | ARABIC LETTER MEEM |
| `U+0646` | `ن` | 27380 | ARABIC LETTER NOON |
| `U+0647` | `ه` | 14962 | ARABIC LETTER HEH |
| `U+0648` | `و` | 24970 | ARABIC LETTER WAW |
| `U+0649` | `ى` | 6603 | ARABIC LETTER ALEF MAKSURA |
| `U+064A` | `ي` | 18334 | ARABIC LETTER YEH |
| `U+064B` | `ً` | 3741 | ARABIC FATHATAN |
| `U+064C` | `ٌ` | 2519 | ARABIC DAMMATAN |
| `U+064D` | `ٍ` | 2633 | ARABIC KASRATAN |
| `U+064E` | `َ` | 123396 | ARABIC FATHA |
| `U+064F` | `ُ` | 37320 | ARABIC DAMMA |
| `U+0650` | `ِ` | 46642 | ARABIC KASRA |
| `U+0651` | `ّ` | 23016 | ARABIC SHADDA |
| `U+0652` | `ْ` | 37372 | ARABIC SUKUN |
| `U+0653` | `ٓ` | 5376 | ARABIC MADDAH ABOVE |
| `U+0654` | `ٔ` | 773 | ARABIC HAMZA ABOVE |
| `U+0670` | `ٰ` | 9838 | ARABIC LETTER SUPERSCRIPT ALEF |
| `U+0671` | `ٱ` | 13819 | ARABIC LETTER ALEF WASLA |
| `U+06DC` | `ۜ` | 2 | ARABIC SMALL HIGH SEEN |
| `U+06DF` | `۟` | 3988 | ARABIC SMALL HIGH ROUNDED ZERO |
| `U+06E0` | `۠` | 66 | ARABIC SMALL HIGH UPRIGHT RECTANGULAR ZERO |
| `U+06E2` | `ۢ` | 510 | ARABIC SMALL HIGH MEEM ISOLATED FORM |
| `U+06E3` | `ۣ` | 1 | ARABIC SMALL LOW SEEN |
| `U+06E5` | `ۥ` | 1257 | ARABIC SMALL WAW |
| `U+06E6` | `ۦ` | 957 | ARABIC SMALL YEH |
| `U+06E7` | `ۧ` | 38 | ARABIC SMALL HIGH YEH |
| `U+06E8` | `ۨ` | 1 | ARABIC SMALL HIGH NOON |
| `U+06EA` | `۪` | 1 | ARABIC EMPTY CENTRE LOW STOP |
| `U+06EB` | `۫` | 1 | ARABIC EMPTY CENTRE HIGH STOP |
| `U+06EC` | `۬` | 1 | ARABIC ROUNDED HIGH STOP WITH FILLED CENTRE |
| `U+06ED` | `ۭ` | 99 | ARABIC SMALL LOW MEEM |

### uthmani-min (56 distinct codepoints)

| Codepoint | Char | Count | Unicode Name |
| :--- | :---: | :--- | :--- |
| `U+0020` | `[SPACE]` | 71645 | SPACE |
| `U+0621` | `ء` | 2783 | ARABIC LETTER HAMZA |
| `U+0623` | `أ` | 8901 | ARABIC LETTER ALEF WITH HAMZA ABOVE |
| `U+0624` | `ؤ` | 706 | ARABIC LETTER WAW WITH HAMZA ABOVE |
| `U+0625` | `إ` | 5088 | ARABIC LETTER ALEF WITH HAMZA BELOW |
| `U+0626` | `ئ` | 921 | ARABIC LETTER YEH WITH HAMZA ABOVE |
| `U+0627` | `ا` | 39002 | ARABIC LETTER ALEF |
| `U+0628` | `ب` | 11603 | ARABIC LETTER BEH |
| `U+0629` | `ة` | 2344 | ARABIC LETTER TEH MARBUTA |
| `U+062A` | `ت` | 10520 | ARABIC LETTER TEH |
| `U+062B` | `ث` | 1414 | ARABIC LETTER THEH |
| `U+062C` | `ج` | 3317 | ARABIC LETTER JEEM |
| `U+062D` | `ح` | 4364 | ARABIC LETTER HAH |
| `U+062E` | `خ` | 2497 | ARABIC LETTER KHAH |
| `U+062F` | `د` | 5991 | ARABIC LETTER DAL |
| `U+0630` | `ذ` | 4932 | ARABIC LETTER THAL |
| `U+0631` | `ر` | 12627 | ARABIC LETTER REH |
| `U+0632` | `ز` | 1599 | ARABIC LETTER ZAIN |
| `U+0633` | `س` | 6122 | ARABIC LETTER SEEN |
| `U+0634` | `ش` | 2124 | ARABIC LETTER SHEEN |
| `U+0635` | `ص` | 2074 | ARABIC LETTER SAD |
| `U+0636` | `ض` | 1686 | ARABIC LETTER DAD |
| `U+0637` | `ط` | 1273 | ARABIC LETTER TAH |
| `U+0638` | `ظ` | 853 | ARABIC LETTER ZAH |
| `U+0639` | `ع` | 9405 | ARABIC LETTER AIN |
| `U+063A` | `غ` | 1221 | ARABIC LETTER GHAIN |
| `U+0640` | `ـ` | 812 | ARABIC TATWEEL |
| `U+0641` | `ف` | 8747 | ARABIC LETTER FEH |
| `U+0642` | `ق` | 7034 | ARABIC LETTER QAF |
| `U+0643` | `ك` | 10497 | ARABIC LETTER KAF |
| `U+0644` | `ل` | 38550 | ARABIC LETTER LAM |
| `U+0645` | `م` | 27071 | ARABIC LETTER MEEM |
| `U+0646` | `ن` | 27380 | ARABIC LETTER NOON |
| `U+0647` | `ه` | 14962 | ARABIC LETTER HEH |
| `U+0648` | `و` | 24970 | ARABIC LETTER WAW |
| `U+0649` | `ى` | 6582 | ARABIC LETTER ALEF MAKSURA |
| `U+064A` | `ي` | 18355 | ARABIC LETTER YEH |
| `U+064B` | `ً` | 3741 | ARABIC FATHATAN |
| `U+064C` | `ٌ` | 2519 | ARABIC DAMMATAN |
| `U+064D` | `ٍ` | 2633 | ARABIC KASRATAN |
| `U+064E` | `َ` | 95999 | ARABIC FATHA |
| `U+064F` | `ُ` | 28052 | ARABIC DAMMA |
| `U+0650` | `ِ` | 37457 | ARABIC KASRA |
| `U+0651` | `ّ` | 19249 | ARABIC SHADDA |
| `U+0654` | `ٔ` | 772 | ARABIC HAMZA ABOVE |
| `U+0670` | `ٰ` | 9837 | ARABIC LETTER SUPERSCRIPT ALEF |
| `U+06DC` | `ۜ` | 2 | ARABIC SMALL HIGH SEEN |
| `U+06DF` | `۟` | 92 | ARABIC SMALL HIGH ROUNDED ZERO |
| `U+06E0` | `۠` | 66 | ARABIC SMALL HIGH UPRIGHT RECTANGULAR ZERO |
| `U+06E3` | `ۣ` | 1 | ARABIC SMALL LOW SEEN |
| `U+06E5` | `ۥ` | 29 | ARABIC SMALL WAW |
| `U+06E6` | `ۦ` | 22 | ARABIC SMALL YEH |
| `U+06E7` | `ۧ` | 38 | ARABIC SMALL HIGH YEH |
| `U+06E8` | `ۨ` | 1 | ARABIC SMALL HIGH NOON |
| `U+06EA` | `۪` | 1 | ARABIC EMPTY CENTRE LOW STOP |
| `U+06EB` | `۫` | 1 | ARABIC EMPTY CENTRE HIGH STOP |

### simple (46 distinct codepoints)

| Codepoint | Char | Count | Unicode Name |
| :--- | :---: | :--- | :--- |
| `U+0020` | `[SPACE]` | 72012 | SPACE |
| `U+0621` | `ء` | 1578 | ARABIC LETTER HAMZA |
| `U+0622` | `آ` | 1511 | ARABIC LETTER ALEF WITH MADDA ABOVE |
| `U+0623` | `أ` | 9119 | ARABIC LETTER ALEF WITH HAMZA ABOVE |
| `U+0624` | `ؤ` | 673 | ARABIC LETTER WAW WITH HAMZA ABOVE |
| `U+0625` | `إ` | 5108 | ARABIC LETTER ALEF WITH HAMZA BELOW |
| `U+0626` | `ئ` | 1182 | ARABIC LETTER YEH WITH HAMZA ABOVE |
| `U+0627` | `ا` | 43875 | ARABIC LETTER ALEF |
| `U+0628` | `ب` | 11603 | ARABIC LETTER BEH |
| `U+0629` | `ة` | 2344 | ARABIC LETTER TEH MARBUTA |
| `U+062A` | `ت` | 10520 | ARABIC LETTER TEH |
| `U+062B` | `ث` | 1414 | ARABIC LETTER THEH |
| `U+062C` | `ج` | 3317 | ARABIC LETTER JEEM |
| `U+062D` | `ح` | 4364 | ARABIC LETTER HAH |
| `U+062E` | `خ` | 2497 | ARABIC LETTER KHAH |
| `U+062F` | `د` | 5991 | ARABIC LETTER DAL |
| `U+0630` | `ذ` | 4932 | ARABIC LETTER THAL |
| `U+0631` | `ر` | 12627 | ARABIC LETTER REH |
| `U+0632` | `ز` | 1599 | ARABIC LETTER ZAIN |
| `U+0633` | `س` | 6124 | ARABIC LETTER SEEN |
| `U+0634` | `ش` | 2124 | ARABIC LETTER SHEEN |
| `U+0635` | `ص` | 2072 | ARABIC LETTER SAD |
| `U+0636` | `ض` | 1686 | ARABIC LETTER DAD |
| `U+0637` | `ط` | 1273 | ARABIC LETTER TAH |
| `U+0638` | `ظ` | 853 | ARABIC LETTER ZAH |
| `U+0639` | `ع` | 9405 | ARABIC LETTER AIN |
| `U+063A` | `غ` | 1221 | ARABIC LETTER GHAIN |
| `U+0641` | `ف` | 8747 | ARABIC LETTER FEH |
| `U+0642` | `ق` | 7034 | ARABIC LETTER QAF |
| `U+0643` | `ك` | 10497 | ARABIC LETTER KAF |
| `U+0644` | `ل` | 38639 | ARABIC LETTER LAM |
| `U+0645` | `م` | 27071 | ARABIC LETTER MEEM |
| `U+0646` | `ن` | 27382 | ARABIC LETTER NOON |
| `U+0647` | `ه` | 14962 | ARABIC LETTER HEH |
| `U+0648` | `و` | 24813 | ARABIC LETTER WAW |
| `U+0649` | `ى` | 2595 | ARABIC LETTER ALEF MAKSURA |
| `U+064A` | `ي` | 22085 | ARABIC LETTER YEH |
| `U+064B` | `ً` | 3742 | ARABIC FATHATAN |
| `U+064C` | `ٌ` | 2519 | ARABIC DAMMATAN |
| `U+064D` | `ٍ` | 2633 | ARABIC KASRATAN |
| `U+064E` | `َ` | 121886 | ARABIC FATHA |
| `U+064F` | `ُ` | 37320 | ARABIC DAMMA |
| `U+0650` | `ِ` | 46642 | ARABIC KASRA |
| `U+0651` | `ّ` | 23016 | ARABIC SHADDA |
| `U+0652` | `ْ` | 37372 | ARABIC SUKUN |
| `U+0670` | `ٰ` | 3330 | ARABIC LETTER SUPERSCRIPT ALEF |

### simple-clean (37 distinct codepoints)

| Codepoint | Char | Count | Unicode Name |
| :--- | :---: | :--- | :--- |
| `U+0020` | `[SPACE]` | 72012 | SPACE |
| `U+0621` | `ء` | 1578 | ARABIC LETTER HAMZA |
| `U+0622` | `آ` | 1511 | ARABIC LETTER ALEF WITH MADDA ABOVE |
| `U+0623` | `أ` | 9119 | ARABIC LETTER ALEF WITH HAMZA ABOVE |
| `U+0624` | `ؤ` | 673 | ARABIC LETTER WAW WITH HAMZA ABOVE |
| `U+0625` | `إ` | 5108 | ARABIC LETTER ALEF WITH HAMZA BELOW |
| `U+0626` | `ئ` | 1182 | ARABIC LETTER YEH WITH HAMZA ABOVE |
| `U+0627` | `ا` | 43875 | ARABIC LETTER ALEF |
| `U+0628` | `ب` | 11603 | ARABIC LETTER BEH |
| `U+0629` | `ة` | 2344 | ARABIC LETTER TEH MARBUTA |
| `U+062A` | `ت` | 10520 | ARABIC LETTER TEH |
| `U+062B` | `ث` | 1414 | ARABIC LETTER THEH |
| `U+062C` | `ج` | 3317 | ARABIC LETTER JEEM |
| `U+062D` | `ح` | 4364 | ARABIC LETTER HAH |
| `U+062E` | `خ` | 2497 | ARABIC LETTER KHAH |
| `U+062F` | `د` | 5991 | ARABIC LETTER DAL |
| `U+0630` | `ذ` | 4932 | ARABIC LETTER THAL |
| `U+0631` | `ر` | 12627 | ARABIC LETTER REH |
| `U+0632` | `ز` | 1599 | ARABIC LETTER ZAIN |
| `U+0633` | `س` | 6124 | ARABIC LETTER SEEN |
| `U+0634` | `ش` | 2124 | ARABIC LETTER SHEEN |
| `U+0635` | `ص` | 2072 | ARABIC LETTER SAD |
| `U+0636` | `ض` | 1686 | ARABIC LETTER DAD |
| `U+0637` | `ط` | 1273 | ARABIC LETTER TAH |
| `U+0638` | `ظ` | 853 | ARABIC LETTER ZAH |
| `U+0639` | `ع` | 9405 | ARABIC LETTER AIN |
| `U+063A` | `غ` | 1221 | ARABIC LETTER GHAIN |
| `U+0641` | `ف` | 8747 | ARABIC LETTER FEH |
| `U+0642` | `ق` | 7034 | ARABIC LETTER QAF |
| `U+0643` | `ك` | 10497 | ARABIC LETTER KAF |
| `U+0644` | `ل` | 38639 | ARABIC LETTER LAM |
| `U+0645` | `م` | 27071 | ARABIC LETTER MEEM |
| `U+0646` | `ن` | 27382 | ARABIC LETTER NOON |
| `U+0647` | `ه` | 14962 | ARABIC LETTER HEH |
| `U+0648` | `و` | 24813 | ARABIC LETTER WAW |
| `U+0649` | `ى` | 2595 | ARABIC LETTER ALEF MAKSURA |
| `U+064A` | `ي` | 22085 | ARABIC LETTER YEH |

### simple-min (45 distinct codepoints)

| Codepoint | Char | Count | Unicode Name |
| :--- | :---: | :--- | :--- |
| `U+0020` | `[SPACE]` | 72012 | SPACE |
| `U+0621` | `ء` | 1578 | ARABIC LETTER HAMZA |
| `U+0622` | `آ` | 1511 | ARABIC LETTER ALEF WITH MADDA ABOVE |
| `U+0623` | `أ` | 9119 | ARABIC LETTER ALEF WITH HAMZA ABOVE |
| `U+0624` | `ؤ` | 673 | ARABIC LETTER WAW WITH HAMZA ABOVE |
| `U+0625` | `إ` | 5108 | ARABIC LETTER ALEF WITH HAMZA BELOW |
| `U+0626` | `ئ` | 1182 | ARABIC LETTER YEH WITH HAMZA ABOVE |
| `U+0627` | `ا` | 43875 | ARABIC LETTER ALEF |
| `U+0628` | `ب` | 11603 | ARABIC LETTER BEH |
| `U+0629` | `ة` | 2344 | ARABIC LETTER TEH MARBUTA |
| `U+062A` | `ت` | 10520 | ARABIC LETTER TEH |
| `U+062B` | `ث` | 1414 | ARABIC LETTER THEH |
| `U+062C` | `ج` | 3317 | ARABIC LETTER JEEM |
| `U+062D` | `ح` | 4364 | ARABIC LETTER HAH |
| `U+062E` | `خ` | 2497 | ARABIC LETTER KHAH |
| `U+062F` | `د` | 5991 | ARABIC LETTER DAL |
| `U+0630` | `ذ` | 4932 | ARABIC LETTER THAL |
| `U+0631` | `ر` | 12627 | ARABIC LETTER REH |
| `U+0632` | `ز` | 1599 | ARABIC LETTER ZAIN |
| `U+0633` | `س` | 6124 | ARABIC LETTER SEEN |
| `U+0634` | `ش` | 2124 | ARABIC LETTER SHEEN |
| `U+0635` | `ص` | 2072 | ARABIC LETTER SAD |
| `U+0636` | `ض` | 1686 | ARABIC LETTER DAD |
| `U+0637` | `ط` | 1273 | ARABIC LETTER TAH |
| `U+0638` | `ظ` | 853 | ARABIC LETTER ZAH |
| `U+0639` | `ع` | 9405 | ARABIC LETTER AIN |
| `U+063A` | `غ` | 1221 | ARABIC LETTER GHAIN |
| `U+0641` | `ف` | 8747 | ARABIC LETTER FEH |
| `U+0642` | `ق` | 7034 | ARABIC LETTER QAF |
| `U+0643` | `ك` | 10497 | ARABIC LETTER KAF |
| `U+0644` | `ل` | 38639 | ARABIC LETTER LAM |
| `U+0645` | `م` | 27071 | ARABIC LETTER MEEM |
| `U+0646` | `ن` | 27382 | ARABIC LETTER NOON |
| `U+0647` | `ه` | 14962 | ARABIC LETTER HEH |
| `U+0648` | `و` | 24813 | ARABIC LETTER WAW |
| `U+0649` | `ى` | 2595 | ARABIC LETTER ALEF MAKSURA |
| `U+064A` | `ي` | 22085 | ARABIC LETTER YEH |
| `U+064B` | `ً` | 3742 | ARABIC FATHATAN |
| `U+064C` | `ٌ` | 2519 | ARABIC DAMMATAN |
| `U+064D` | `ٍ` | 2633 | ARABIC KASRATAN |
| `U+064E` | `َ` | 96040 | ARABIC FATHA |
| `U+064F` | `ُ` | 28052 | ARABIC DAMMA |
| `U+0650` | `ِ` | 37457 | ARABIC KASRA |
| `U+0651` | `ّ` | 19248 | ARABIC SHADDA |
| `U+0670` | `ٰ` | 3330 | ARABIC LETTER SUPERSCRIPT ALEF |

## 4. Continuous Word Stream Window Simulation

| Words/Line | Window (3 Lines) | Cross-Ayah Share | Ayahs < 1 Line | Line Chars (med / p95 / max) |
| :---: | :---: | :---: | :---: | :---: |
| 4 | 12 words | 81.1% | 549 (8.8%) | 20.0 / 26.0 / 38 |
| 6 | 18 words | 90.6% | 1502 (24.1%) | 30.0 / 38.0 / 59 |
| 8 | 24 words | 95.4% | 2208 (35.4%) | 41.0 / 49.0 / 81 |
