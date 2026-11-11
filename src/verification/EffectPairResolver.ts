/**
 * EffectPairResolver - Discover and resolve HLSL/WGSL effect pairs.
 * Scans effects_src and effects directories for .hlsl and .wgsl files,
 * matches by relative path and basename to find corresponding pairs for verification.
 */

import { glob } from 'glob';
import { resolve, relative, basename, dirname } from 'path';
import { fileURLToPath } from 'url';

export interface EffectPair {
    name: string;
    hlslPath: string;
    wgslPath: string;
    category: string;
    hlslRelativePath: string;
    wgslRelativePath: string;
}

export interface ResolverOptions {
    hlslRoot?: string;
    wgslRoot?: string;
    workspaceRoot?: string;
}

export class EffectPairResolver {
    private hlslRoot: string;
    private wgslRoot: string;
    private workspaceRoot: string;

    constructor(options: ResolverOptions = {}) {
        this.workspaceRoot = options.workspaceRoot || process.cwd();
        this.hlslRoot = resolve(this.workspaceRoot, options.hlslRoot || 'effects_src');
        this.wgslRoot = resolve(this.workspaceRoot, options.wgslRoot || 'effects');
    }

    async discover(): Promise<EffectPair[]> {
        const [hlslFiles, wgslFiles] = await Promise.all([
            this.globFiles(this.hlslRoot, '**/*.hlsl'),
            this.globFiles(this.wgslRoot, '**/*.wgsl'),
        ]);

        // Build multiple lookup maps for WGSL files
        const wgslByCategoryName = new Map<string, string>(); // "category/name" -> path
        const wgslByName = new Map<string, string[]>(); // "name" -> [paths] (for cross-category match)
        
        for (const wgslPath of wgslFiles) {
            const relPath = relative(this.wgslRoot, wgslPath);
            const category = dirname(relPath).split('/')[0] || 'root';
            const name = basename(wgslPath, '.wgsl');
            
            const categoryKey = `${category}/${name}`;
            wgslByCategoryName.set(categoryKey, wgslPath);
            
            if (!wgslByName.has(name)) {
                wgslByName.set(name, []);
            }
            wgslByName.get(name)!.push(wgslPath);
        }

        const pairs: EffectPair[] = [];
        for (const hlslPath of hlslFiles) {
            const relPath = relative(this.hlslRoot, hlslPath);
            const hlslCategory = dirname(relPath).split('/')[0] || 'root';
            const name = basename(hlslPath, '.hlsl');
            
            // Try exact category match first
            const categoryKey = `${hlslCategory}/${name}`;
            let wgslPath = wgslByCategoryName.get(categoryKey);
            let matchedCategory = hlslCategory;
            
            // If no exact match, try cross-category match (e.g., HLSL at root, WGSL in Scalers/)
            if (!wgslPath) {
                const candidates = wgslByName.get(name) || [];
                if (candidates.length === 1) {
                    // Unique name across all categories - use it
                    wgslPath = candidates[0];
                    matchedCategory = dirname(relative(this.wgslRoot, wgslPath)).split('/')[0] || 'root';
                } else if (candidates.length > 1) {
                    // Multiple candidates - prefer one with matching category if possible
                    for (const candidate of candidates) {
                        const candCategory = dirname(relative(this.wgslRoot, candidate)).split('/')[0] || 'root';
                        if (candCategory === hlslCategory) {
                            wgslPath = candidate;
                            matchedCategory = candCategory;
                            break;
                        }
                    }
                    // If still no match, use first candidate
                    if (!wgslPath) {
                        wgslPath = candidates[0];
                        matchedCategory = dirname(relative(this.wgslRoot, wgslPath)).split('/')[0] || 'root';
                    }
                }
            }
            
            if (wgslPath) {
                pairs.push({
                    name,
                    hlslPath,
                    wgslPath,
                    category: matchedCategory,
                    hlslRelativePath: relPath,
                    wgslRelativePath: relative(this.wgslRoot, wgslPath),
                });
            }
        }

        pairs.sort((a, b) => {
            const catCmp = a.category.localeCompare(b.category);
            if (catCmp !== 0) return catCmp;
            return a.name.localeCompare(b.name);
        });

        return pairs;
    }

    async resolveByName(name: string, category?: string): Promise<EffectPair | null> {
        const pairs = await this.discover();
        const lowerName = name.toLowerCase();
        const lowerCategory = category?.toLowerCase();

        // Exact match first
        for (const pair of pairs) {
            if (pair.name.toLowerCase() === lowerName &&
                (!lowerCategory || pair.category.toLowerCase() === lowerCategory)) {
                return pair;
            }
        }

        // Fuzzy match: contains
        for (const pair of pairs) {
            if (pair.name.toLowerCase().includes(lowerName) &&
                (!lowerCategory || pair.category.toLowerCase() === lowerCategory)) {
                return pair;
            }
        }

        return null;
    }

    async listCategories(): Promise<string[]> {
        const pairs = await this.discover();
        const categories = new Set(pairs.map(p => p.category));
        return Array.from(categories).sort();
    }

    async listEffects(category?: string): Promise<string[]> {
        const pairs = await this.discover();
        const filtered = category
            ? pairs.filter(p => p.category.toLowerCase() === category.toLowerCase())
            : pairs;
        return filtered.map(p => p.name);
    }

    async getByCategory(category: string): Promise<EffectPair[]> {
        const pairs = await this.discover();
        return pairs.filter(p => p.category.toLowerCase() === category.toLowerCase());
    }

    private async globFiles(root: string, pattern: string): Promise<string[]> {
        return glob(pattern, { cwd: root, absolute: true, nodir: true });
    }
}

export function createResolver(options?: ResolverOptions): EffectPairResolver {
    return new EffectPairResolver(options);
}