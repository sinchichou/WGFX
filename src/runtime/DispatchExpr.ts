/// <reference types="@webgpu/types" />

/**
 * Dispatch expression evaluator — CSP-safe integer arithmetic for `//! DISPATCH`.
 * ---
 * `//! DISPATCH` 運算式的整數算術求值器 —— **不使用 eval / new Function**。
 *
 * 為什麼不用 `new Function` / `eval`
 * ---
 * Chrome MV3 擴充套件的預設 Content Security Policy 為
 * `script-src 'self'; upgrade-insecure-requests;`，明確禁止將字串當程式碼求值。
 * WGFX 是要在瀏覽器擴充套件中執行的核心庫，因此 `//! DISPATCH` 的求值**必須**
 * 不依賴動態程式碼生成。
 *
 * `isolated_world` CSP（可允許 `unsafe-eval`）是 Chromium 的實驗性功能，需要自行
 * 編譯瀏覽器才能使用；本專案不依賴它。
 *
 * 設計
 * ---
 * 以**遞迴下降解析**（recursive-descent parser）把運算式解析成 AST，
 * 再以純資料流求值。求值過程只做整數運算，不涉及任何字串 → 程式碼的轉換。
 *
 * 支援的文法
 * ---
 * ```text
 * expr    := term (('+' | '-') term)*
 * term    := factor (('*' | '/' | '%') factor)*
 * factor  := unary (('<<' | '>>' | '>>>' | '&' | '|' | '^') unary)*
 * unary   := ('-' | '~' | '+')? primary
 * primary := NUMBER | IDENT | '(' expr ')'
 * ```
 *
 * 語意與 JavaScript 對齊的部分：
 * - `/` 為浮點除法（與 JS 一致），最終結果再由呼叫端取整
 * - `>>` 為有號右移、`>>>` 為無號右移
 * - 運算元為 32 位元有號整數（`ToInt32` 語意），超出範圍會回繞
 *
 * @group Core
 * @category Internal
 */

/** 求值器的變數上下文（變數名 → 數值）。 */
export type DispatchContext = Record<string, number>;

/** AST 節點。刻意設計為純資料結構，便於快取與重用。 */
export type DispatchNode =
    | { kind: 'num'; value: number }
    | { kind: 'var'; name: string }
    | { kind: 'unary'; op: '-' | '~' | '+'; operand: DispatchNode }
    | {
          kind: 'binary';
          op: '+' | '-' | '*' | '/' | '%' | '<<' | '>>' | '>>>' | '&' | '|' | '^';
          left: DispatchNode;
          right: DispatchNode;
      };

/** 剖析錯誤（帶位置資訊，便於除錯）。 */
export class DispatchExprError extends Error {
    public readonly position: number;
    public readonly expression: string;

    constructor(message: string, expression: string, position: number) {
        super(message);
        this.name = 'DispatchExprError';
        this.expression = expression;
        this.position = position;
    }
}

/**
 * ToInt32 — 將值轉為 32 位元有號整數（與 JavaScript 的 `| 0` 語意相同）。
 */
function toInt32(value: number): number {
    if (!Number.isFinite(value)) return 0;
    const n = Math.trunc(value);
    return n | 0;
}

/**
 * 求值 AST 節點。
 * @internal
 */
function evaluateNode(node: DispatchNode, context: DispatchContext): number {
    switch (node.kind) {
        case 'num':
            return node.value;

        case 'var': {
            const raw = context[node.name];
            if (raw === undefined) {
                throw new DispatchExprError(
                    `Undefined variable: ${node.name}`,
                    node.name,
                    0,
                );
            }
            return toInt32(raw);
        }

        case 'unary': {
            const v = evaluateNode(node.operand, context);
            switch (node.op) {
                case '-': return toInt32(-v);
                case '~': return toInt32(~v);
                case '+': return toInt32(v);
            }
            return 0;
        }

        case 'binary': {
            const left = evaluateNode(node.left, context);
            const right = evaluateNode(node.right, context);
            switch (node.op) {
                case '+': return toInt32(left + right);
                case '-': return toInt32(left - right);
                case '*': return toInt32(left * right);
                // `/` 保持浮點語意（與 JS 一致）；最終取整由呼叫端處理
                case '/': return left / right;
                case '%': return left % right;
                case '<<': return toInt32(left << (right & 31));
                case '>>': return toInt32(left >> (right & 31));
                // 無號右移：先轉成 uint32 再位移
                case '>>>': return toInt32((left >>> 0) >>> (right & 31));
                case '&': return toInt32(left & right);
                case '|': return toInt32(left | right);
                case '^': return toInt32(left ^ right);
            }
            return 0;
        }
    }
}

/**
 * 一個已剖析的運算式，可依不同上下文重複求值。
 * ---
 * 剖析只做一次（建置管線時），求值是純資料流運算 —— 這讓每幀的 dispatch 計算
 * 既不需要字串 → 程式碼的轉換，也不需重新剖析。
 */
export class DispatchExpression {
    /** 原始運算式文字（除錯與錯誤訊息用）。 */
    public readonly source: string;
    /** 剖析出的 AST。 */
    public readonly ast: DispatchNode;

    constructor(source: string, ast: DispatchNode) {
        this.source = source;
        this.ast = ast;
    }

    /**
     * 以給定上下文求值。
     * @param context - 變數名 → 數值
     */
    public evaluate(context: DispatchContext): number {
        return evaluateNode(this.ast, context);
    }
}

/**
 * Tokenizer — 將運算式拆成 token。
 * @internal
 */
type TokenType = 'num' | 'ident' | 'op' | 'lparen' | 'rparen';
interface Token {
    type: TokenType;
    value: string;
    pos: number;
}

/** 兩字元運算子（必須先於單字元比對）。 */
const TWO_CHAR_OPS = ['<<', '>>', '>>>'];
/** 單字元運算子。 */
const ONE_CHAR_OPS = ['+', '-', '*', '/', '%', '&', '|', '^', '~'];

function tokenize(source: string): Token[] {
    const tokens: Token[] = [];
    let i = 0;
    while (i < source.length) {
        const ch = source[i];

        // 空白
        if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') {
            i++;
            continue;
        }

        // 數字（整數或小數）
        if (ch >= '0' && ch <= '9') {
            const start = i;
            while (i < source.length && source[i] >= '0' && source[i] <= '9') i++;
            if (source[i] === '.') {
                i++;
                while (i < source.length && source[i] >= '0' && source[i] <= '9') i++;
            }
            tokens.push({ type: 'num', value: source.slice(start, i), pos: start });
            continue;
        }

        // 識別字
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_') {
            const start = i;
            while (i < source.length && /[A-Za-z0-9_]/.test(source[i])) i++;
            tokens.push({ type: 'ident', value: source.slice(start, i), pos: start });
            continue;
        }

        // 左括號
        if (ch === '(') {
            tokens.push({ type: 'lparen', value: ch, pos: i });
            i++;
            continue;
        }

        // 右括號
        if (ch === ')') {
            tokens.push({ type: 'rparen', value: ch, pos: i });
            i++;
            continue;
        }

        // 兩字元運算子
        const two = source.slice(i, i + 3);
        if (TWO_CHAR_OPS.includes(two)) {
            tokens.push({ type: 'op', value: two, pos: i });
            i += two.length;
            continue;
        }
        const two2 = source.slice(i, i + 2);
        if (TWO_CHAR_OPS.includes(two2)) {
            tokens.push({ type: 'op', value: two2, pos: i });
            i += 2;
            continue;
        }

        // 單字元運算子
        if (ONE_CHAR_OPS.includes(ch)) {
            tokens.push({ type: 'op', value: ch, pos: i });
            i++;
            continue;
        }

        throw new DispatchExprError(
            `Unexpected character "${ch}"`,
            source,
            i,
        );
    }
    return tokens;
}

/**
 * Parser — 遞迴下降剖析。
 * @internal
 */
class Parser {
    private tokens: Token[];
    private index = 0;
    private source: string;

    constructor(source: string) {
        this.source = source;
        this.tokens = tokenize(source);
    }

    private peek(): Token | undefined {
        return this.tokens[this.index];
    }

    private next(): Token | undefined {
        return this.tokens[this.index++];
    }

    private expect(type: TokenType, what: string): Token {
        const t = this.next();
        if (!t || t.type !== type) {
            throw new DispatchExprError(
                `Expected ${what} but found ${t ? `"${t.value}"` : 'end of expression'}`,
                this.source,
                t ? t.pos : this.source.length,
            );
        }
        return t;
    }

    /** expr := term (('+' | '-') term)* */
    public parseExpression(): DispatchNode {
        let left = this.parseTerm();
        for (;;) {
            const t = this.peek();
            if (!t || t.type !== 'op' || (t.value !== '+' && t.value !== '-')) break;
            this.next();
            const right = this.parseTerm();
            left = { kind: 'binary', op: t.value as '+' | '-', left, right };
        }
        return left;
    }

    /** term := factor (('*' | '/' | '%') factor)* */
    private parseTerm(): DispatchNode {
        let left = this.parseFactor();
        for (;;) {
            const t = this.peek();
            if (!t || t.type !== 'op' || !['*', '/', '%'].includes(t.value)) break;
            this.next();
            const right = this.parseFactor();
            left = {
                kind: 'binary',
                op: t.value as '*' | '/' | '%',
                left,
                right,
            };
        }
        return left;
    }

    /** factor := unary (('<<' | '>>' | '>>>' | '&' | '|' | '^') unary)* */
    private parseFactor(): DispatchNode {
        let left = this.parseUnary();
        for (;;) {
            const t = this.peek();
            if (!t || t.type !== 'op') break;
            if (!['<<', '>>', '>>>', '&', '|', '^'].includes(t.value)) break;
            this.next();
            const right = this.parseUnary();
            left = {
                kind: 'binary',
                op: t.value as '<<' | '>>' | '>>>' | '&' | '|' | '^',
                left,
                right,
            };
        }
        return left;
    }

    /** unary := ('-' | '~' | '+')? primary */
    private parseUnary(): DispatchNode {
        const t = this.peek();
        if (t && t.type === 'op' && (t.value === '-' || t.value === '~' || t.value === '+')) {
            this.next();
            const operand = this.parseUnary();
            return { kind: 'unary', op: t.value as '-' | '~' | '+', operand };
        }
        return this.parsePrimary();
    }

    /** primary := NUMBER | IDENT | '(' expr ')' */
    private parsePrimary(): DispatchNode {
        const t = this.next();
        if (!t) {
            throw new DispatchExprError('Unexpected end of expression', this.source, this.source.length);
        }
        if (t.type === 'num') {
            return { kind: 'num', value: Number(t.value) };
        }
        if (t.type === 'ident') {
            return { kind: 'var', name: t.value };
        }
        if (t.type === 'lparen') {
            const inner = this.parseExpression();
            this.expect('rparen', '")"');
            return inner;
        }
        throw new DispatchExprError(
            `Unexpected token "${t.value}"`,
            this.source,
            t.pos,
        );
    }

    /** 剖析完成後檢查是否有多餘 token。 */
    public assertFullyConsumed(): void {
        const t = this.peek();
        if (t) {
            throw new DispatchExprError(
                `Unexpected token "${t.value}" after expression`,
                this.source,
                t.pos,
            );
        }
    }
}

/**
 * 剖析一個 `//! DISPATCH` 運算式。
 * ---
 * 剖析只應在**建置管線時**做一次；每幀以 {@link DispatchExpression.evaluate} 求值。
 *
 * @param source - 運算式文字，例如 `(((NW>>1)*(NH>>1)/4)*8+63)/64`
 * @returns 可重複求值的 {@link DispatchExpression}
 * @throws {DispatchExprError} 語法錯誤或出現未定義的變數
 */
export function parseDispatchExpression(source: string): DispatchExpression {
    const parser = new Parser(source);
    const ast = parser.parseExpression();
    parser.assertFullyConsumed();
    return new DispatchExpression(source, ast);
}

/**
 * 一次性求值（剖析 + 求值）。便利函式，主要用於測試。
 */
export function evaluateDispatchExpression(source: string, context: DispatchContext): number {
    return parseDispatchExpression(source).evaluate(context);
}
