/*
    A C implementation of a red-black tree memory allocator.
    Copyright (C) 2023 Clayton Ramsey.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * A red-black tree memory allocator implementation for the CS:APP `malloc` assignment.
 *
 * In this approach, all free blocks are nodes in one large red-black tree.
 * To find a free node to allocate, we simply search the red-black tree in logarithmic time
 * complexity for its best possible fit.
 *
 * Additionally, this solution implements free block splitting, coalescing, and efficient
 * reallocation of the rightmost-allocated block.
 *
 * This allocator is not thread-safe; it must only be called via mutual exclustion or in a single-
 * threaded context.
 *
 * Conditional compilation options are available:
 * - To include sanitization checks, compile with `DEBUG` defined.
 * - To include printouts on allocation, compile with `VERBOSE` defined.
 */

#include <assert.h>
#include <stdbool.h>
#include <string.h>

/**
 * Includes for CS:APP libraries.
 * Their content is not included in this repository due to copyright.
 */
#include "memlib.h"
#include "mm.h"

#ifdef VERBOSE
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

#ifdef DEBUG
#define DEBUG_ASSERT(x) assert(x)
#else
#define DEBUG_ASSERT(x)
#endif

#ifdef DEBUG
#define DEBUG_EXECUTE(x) x
#else
#define DEBUG_EXECUTE(x)
#endif

/* Data structures */

/**
 * The flags of a block.
 * These contain packed values for a block's size and its state.
 *
 * In order from LSB to MSB, these are the meanings of the values inside a `BlockFlags`:
 * - 1 bit: Whether the block is allocated
 * - 1 bit: The color of the block as a node in the free tree (only for free blocks).
 *   A value of 0 corresponds to a black node while a value of 1 corresponds to a red node.
 * - 1 bit: Unused.
 *   On 32-bit systems, this bit will be used by size.
 * - 61 bits: The size of the block, divided by 8.
 *   On 32-bit systems, the most significant 30 bits are the block size divided by 4.
 *
 * We do not use a bit-packed struct here because it enables us to more cheaply compute information
 * from the flags.
 * For example, if we used a packed struct, we would have to shift the size of each block left
 * by 3 bits every time we wanted to work with a real size.
 */
typedef size_t BlockFlags;

/**
 * The colors that a node in the free tree can be.
 * A `Color` is either `RED` or `BLACK`.
 */
typedef enum __attribute__((__packed__)) color {
    /**
     * A black node, which counts toward the black-height of a red-black tree.
     * A NIL node is also black.
     */
    BLACK = 0,
    /**
     * A red node, which does not count toward the black-height of a red-black tree.
     */
    RED = 2,
} Color;

/**
 * The side that a node can be in a tree.
 */
typedef enum __attribute__((__packed__)) side {
    /**
     * The left side of a node's children, corresponding to the first entry in its children array.
     */
    LEFT = 0,
    /**
     * The right side of a node's children, corresponding to the second entry in its children array.
     */
    RIGHT = 1,
} Side;

/**
 * The header of a free block.
 */
typedef struct fh {
    /**
     * The size of the block, and flags determining its color.
     */
    BlockFlags size_flags;
    /**
     * A pointer to the parent of this block.
     * Will be `NULL` for the root.
     */
    struct fh *parent;
    /**
     * The left and right children of this block.
     *
     * This array should be indexed by `Side`s.
     *
     * If a value in this array is `NULL`, the corresponding child is a NIL node.
     *
     * The left child is guaranteed to have size less than or equal to its parent,
     * while the right child will have size greater than or equal to its parent.
     */
    struct fh *children[2];
} FreeHeader;

/**
 * The header for any block of memory.
 * The block may or may not be allocated.
 * This struct can be "punned" with `FreeHeader` if `size_flags` is marked as unallocated.
 */
typedef struct bh {
    /**
     * The size and flags (packed) of the block.
     */
    BlockFlags size_flags;
    /**
     * The start section of the data block for this header.
     */
    char data_start[0];
} BlockHeader;

/**
 * The footer of a block.
 */
typedef struct {
    /**
     * The size and flags (packed) of the block.
     * This is exactly the same as the header, except for the fact that the block will have no
     * information regarding its color if it is free.
     */
    BlockFlags size_flags;
} Footer;

/**
 * The amount of extra data associated with every block which is not in the payload.
 */
#define BLOCK_OVERHEAD (sizeof(Footer) + sizeof(BlockHeader))

/**
 * The minimum legal size for a payload.
 */
#define MIN_PAYLOAD_SIZE (sizeof(FreeHeader) - sizeof(BlockHeader))

/**
 * The minimum total block size.
 */
#define MIN_BLOCK_SIZE (BLOCK_OVERHEAD + MIN_PAYLOAD_SIZE)

/**
 * The minimum leftover size required to split out a block during `malloc`.
 */
#define MIN_SPLIT_BLOCK_SIZE 3400

/**
 * The size of a pointer.
 */
#define WSIZE (sizeof(void *))

/**
 * The error code returned by `mem_sbrk` on error.
 */
#define SBRK_ERROR ((void *)-1)

/* Static fields */

/**
 * The root of the free block set.
 */
static FreeHeader *rb_root = NULL;

/**
 * The first (lowest-pointer) block that has been allocated.
 */
static BlockHeader *first_block = NULL;

/**
 * The last (highest-pointer) block that has been allocated.
 */
static BlockHeader *last_block = NULL;

/* Helper function declarations */

/**
 * Get the size of a block.
 */
static inline size_t data_size(BlockFlags flags);

/**
 * Determine whether a block is allocated.
 */
static inline bool is_allocated(BlockFlags flags);

/**
 * Find the footer of a block.
 */
static inline Footer *find_footer(const BlockHeader *header);

/**
 * Find the footer of a free block.
 */
static inline Footer *find_footer_free(const FreeHeader *header);

/**
 * Get a pointer to the header of a block based on its data section.
 */
static inline BlockHeader *header_from_data(void *data);

/**
 * Get a pointer to the block immediately to the right of `block` in memory.
 * Should not be called on the last block.
 */
static inline BlockHeader *next_block(const BlockHeader *bh);

/**
 * Find the block immediately after a free block in memory.
 */
static inline BlockHeader *next_block_free(const FreeHeader *header);

/**
 * Set the color of a node to be red.
 */
static inline void mark_red(FreeHeader *node);

/**
 * Set the color of a node to be black.
 */
static inline void mark_black(FreeHeader *node);

/**
 * Determine whether a node is red.
 */
static inline bool is_red(const FreeHeader *node);

/**
 * Determine whether a node is black.
 */
static inline bool is_black(const FreeHeader *node);

/**
 * Add `node` to the red-black tree of free nodes.
 */
static inline void rb_add(FreeHeader *node);

/**
 * Delete `node` from the free tree.
 */
static void rb_delete(FreeHeader *node);

/**
 * Helper function for deleting from red-black trees.
 * `node` must be a black node with two `NULL` children and a parent.
 */
static inline void rb_delete_hard(FreeHeader *node);

/**
 * Rotate `parent` down to be a child of its child on `side`.
 * The child of `parent` opposite `side` must not be `NULL`.
 */
static inline void rb_rotate(FreeHeader *parent, Side side);

/**
 * Get the side that a child node is on given its parent.
 */
static inline Side child_side(const FreeHeader *child);

/**
 * Check that red-black subtree under `node` is valid.
 * Returns the black-depth of the subtree.
 */
static size_t rb_check_subtree(const FreeHeader *node);

/**
 * Get an adjusted (word-aligned) size for an allocation.
 * `request_size` must be greater than 0.
 */
static inline size_t adjust_size(size_t request_size);

/**
 * Check that the heap is valid.
 */
static void check_heap();

static_assert(MIN_BLOCK_SIZE % WSIZE == 0, "minimum block size is not word aligned");
static_assert(MIN_PAYLOAD_SIZE % WSIZE == 0, "minimum payload size is not word aligned");
static_assert(sizeof(BlockHeader) % WSIZE == 0, "block header size is not word aligned");
static_assert(sizeof(FreeHeader) % WSIZE == 0, "free header size is not word aligned");
static_assert(sizeof(Footer) % WSIZE == 0, "footer size is not word aligned");
static_assert(MIN_SPLIT_BLOCK_SIZE % WSIZE == 0, "split-out blocks are not word aligned");
static_assert(MIN_BLOCK_SIZE <= MIN_SPLIT_BLOCK_SIZE, "split-out blocks are too small");

/* Public functions */

int mm_init(void) {
    (void)check_heap;
    (void)next_block_free;

    mem_reset_brk();

    first_block = mem_heap_hi() + 1;
    last_block = NULL;
    rb_root = NULL;

    DEBUG_EXECUTE(check_heap());
    return 0;
}

void *mm_malloc(size_t size) {
    DEBUG_PRINTF("mm_malloc(%zu)\n", size);
    DEBUG_EXECUTE(check_heap());

    if (size == 0)
        return NULL;

    const size_t asize = adjust_size(size);

    // Find the best-fitting node in the tree for this block.

    FreeHeader *best_node = NULL;      // the best-fitting node we've found so far
    size_t free_block_size = SIZE_MAX; // size of the best-fitting node we've found so far

    {
        FreeHeader *node = rb_root;

        while (node != NULL) {
            const size_t n_size = data_size(node->size_flags);
            if (asize < n_size) {
                if (n_size < free_block_size) {
                    best_node = node;
                    free_block_size = n_size;
                }
                node = node->children[LEFT];
            } else if (asize == n_size) {
                best_node = node;
                free_block_size = n_size;
                break;
            } else {
                // asize > node_size
                node = node->children[RIGHT];
            }
        }
    }

    void *to_return; // pointer that will be returned after the giant conditional block below

    if (best_node == NULL) {
        if (last_block == NULL || is_allocated(last_block->size_flags)) {
            // create a new block by expanding the heap
            BlockHeader *const bh = mem_sbrk(BLOCK_OVERHEAD + asize);
            if (bh == SBRK_ERROR)
                return NULL;
            bh->size_flags = asize | 0x1;
            find_footer(bh)->size_flags = bh->size_flags;
            DEBUG_ASSERT(data_size(bh->size_flags) >= asize);
            last_block = bh;
            to_return = &bh->data_start;
        } else {
            // take over the rightmost block in the heap, and then grow it until it has enough space
            // will not overflow since `last_block` was not a fit
            if (mem_sbrk(asize - data_size(last_block->size_flags)) == SBRK_ERROR)
                return NULL;

            rb_delete((FreeHeader *)last_block);
            last_block->size_flags = asize | 0x1;
            find_footer(last_block)->size_flags = last_block->size_flags;
            DEBUG_ASSERT(data_size(last_block->size_flags) >= asize);
            to_return = &last_block->data_start;
        }
    } else {
        // we found a fit
        BlockHeader *const ret_bh = (BlockHeader *)best_node;
        rb_delete(best_node);

        // check if we can split this block
        const size_t leftover_size = free_block_size - asize;
        if (leftover_size >= MIN_SPLIT_BLOCK_SIZE) {
            // we can split the block
            best_node->size_flags = asize;
            // returns a lower header now that we've reduced best_node's size
            BlockHeader *split_header = next_block_free(best_node);
            split_header->size_flags = leftover_size - BLOCK_OVERHEAD;

            // possibly grow this split-out block into its right neighbor
            if (ret_bh == last_block)
                // if we split the last block, the new last block is the one we just created
                last_block = split_header;

            // bookkeeping for the newly-split block, then add the split-out unused block to the
            // tree of free blocks
            find_footer(split_header)->size_flags = split_header->size_flags;
            rb_add((FreeHeader *)split_header);

            DEBUG_ASSERT(next_block(ret_bh) == split_header);
            DEBUG_ASSERT(split_header <= last_block);
        }

        Footer *const ret_footer = find_footer_free(best_node);
        ret_bh->size_flags |= 0b1;
        ret_footer->size_flags = ret_bh->size_flags;
        to_return = &ret_bh->data_start;

        DEBUG_ASSERT(is_allocated(ret_bh->size_flags));
        DEBUG_ASSERT(data_size(ret_bh->size_flags) >= asize);
    }

    DEBUG_ASSERT((void *)&first_block->data_start <= to_return);
    DEBUG_ASSERT(to_return <= (void *)&last_block->data_start);
    DEBUG_ASSERT((char *)to_return + size <= (char *)mem_heap_hi());
    DEBUG_EXECUTE(check_heap());
    return to_return;
}

void mm_free(void *ptr) {
    BlockHeader *const bh = header_from_data(ptr);
    DEBUG_PRINTF("mm_free(%p), [size %zu]\n", ptr, data_size(bh->size_flags));
    DEBUG_EXECUTE(check_heap());

    // mark the block as free
    bh->size_flags &= ~0b11;
    FreeHeader *const fh = (FreeHeader *)bh;
    FreeHeader *final_fh = fh;
    size_t final_size = final_fh->size_flags;

    // attempt to coalesce the block with its neighbors
    // look right
    if (bh != last_block) {
        BlockHeader *const next_bh = next_block_free(fh);
        DEBUG_ASSERT(next_bh->size_flags != 0);
        if (!is_allocated(next_bh->size_flags)) {
            FreeHeader *const next_fh = (FreeHeader *)next_bh;
            rb_delete(next_fh);
            final_size += data_size(next_fh->size_flags) + BLOCK_OVERHEAD;
            // ensure that last blocks are updated correctly
            if (next_bh == last_block)
                last_block = (BlockHeader *)final_fh;
        }
    }

    // look left
    if (bh != first_block) {
        Footer *const prev_bf = (Footer *)((char *)bh - sizeof(Footer));
        if (!is_allocated(prev_bf->size_flags)) {
            FreeHeader *const prev_fh =
                (FreeHeader *)((char *)prev_bf - prev_bf->size_flags - sizeof(BlockHeader));
            rb_delete(prev_fh);
            final_size += data_size(prev_fh->size_flags) + BLOCK_OVERHEAD;
            final_fh = prev_fh;
            if (bh == last_block)
                last_block = (BlockHeader *)final_fh;
        }
    }

    // done coalescing! update the footer to match
    final_fh->size_flags = final_size;
    find_footer_free(final_fh)->size_flags = final_size;

    // bh is now in a state where it can be observed as a free header
    rb_add(final_fh);

    DEBUG_EXECUTE(rb_print_tree(rb_root));
    DEBUG_PRINTF("\n");
    DEBUG_EXECUTE(check_heap());
}

void *mm_realloc(void *ptr, size_t size) {
    DEBUG_PRINTF("mm_realloc(%p, %zu)\n", ptr, size);
    DEBUG_EXECUTE(check_heap());

    // check for spurious reallocations
    if (ptr == NULL)
        return mm_malloc(size);
    if (size == 0)
        return NULL;

    BlockHeader *const bh = header_from_data(ptr);
    const size_t old_size = data_size(bh->size_flags);
    if (size <= old_size)
        // block has enough room for the reallocated size
        return ptr;

    if (bh == last_block) {
        // if this block is the largest block, expand the block by expanding the heap
        const size_t asize = adjust_size(size);
        void *const break_ptr = mem_sbrk(asize - old_size);
        if (break_ptr != SBRK_ERROR) {
            // if sbrk fails, we'll fall through to another case
            bh->size_flags = asize | 0b1;
            find_footer(bh)->size_flags = bh->size_flags;
            DEBUG_EXECUTE(check_heap());
            return ptr;
        }
    }

    // note: theoretically we could also make some optimizations by growing into the block
    // immediately to the right of the reallocated block (saving us a memcpy) but this is not all
    // that important.

    // find a new block which is large enough for us
    void *const new_ptr = mm_malloc(size);
    if (new_ptr == NULL)
        // handle allocation failure by passing it along
        return NULL;
    memcpy(new_ptr, ptr, old_size);

    mm_free(ptr);
    DEBUG_EXECUTE(check_heap());

    return new_ptr;
}

/* Helper function implementations */

static inline size_t data_size(BlockFlags flags) { return flags & ~0b11; }

static inline bool is_allocated(BlockFlags flags) { return flags & 0b01; }

static inline BlockHeader *header_from_data(void *data) {
    return (BlockHeader *)((char *)data - sizeof(BlockHeader));
}

static inline Side child_side(const FreeHeader *const child) {
    DEBUG_ASSERT(child->parent->children[LEFT] == child || child->parent->children[RIGHT] == child);
    return child->parent->children[RIGHT] == child ? RIGHT : LEFT;
}

static inline Footer *find_footer(const BlockHeader *const header) {
    return (Footer *)((char *)header + sizeof(BlockHeader) + data_size(header->size_flags));
}

static inline Footer *find_footer_free(const FreeHeader *const header) {
    return (Footer *)((char *)header + sizeof(BlockHeader) + data_size(header->size_flags));
}

static inline BlockHeader *next_block(const BlockHeader *const header) {
    return (BlockHeader *)((char *)header + sizeof(BlockHeader) + data_size(header->size_flags) +
                           sizeof(Footer));
}

static inline BlockHeader *next_block_free(const FreeHeader *const header) {
    return (BlockHeader *)((char *)header + sizeof(BlockHeader) + data_size(header->size_flags) +
                           sizeof(Footer));
}

static inline void mark_red(FreeHeader *const node) { node->size_flags |= RED; }

static inline void mark_black(FreeHeader *const node) { node->size_flags &= ~RED; }

static inline bool is_red(const FreeHeader *const node) { return node->size_flags & RED; }

static inline bool is_black(const FreeHeader *const node) { return !is_red(node); }

static inline void rb_add(FreeHeader *node) {
    FreeHeader *parent = NULL;
    FreeHeader *next_node = rb_root;
    Side side = LEFT;

    while (next_node != NULL) {
        parent = next_node;
        // it's OK that we don't mask out the flags here because it doesn't affect the true ordering
        // of the tree
        side = node->size_flags <= next_node->size_flags ? LEFT : RIGHT;
        next_node = parent->children[side];
    }

    mark_red(node);
    node->children[LEFT] = NULL;
    node->children[RIGHT] = NULL;
    node->parent = parent;

    // handle empty tree case
    if (parent == NULL) {
        rb_root = node;
        return; // done!
    }

    // parent is now known to be non-null.
    // add ourselves to the parent's children.
    DEBUG_ASSERT(parent->children[side] == NULL);
    parent->children[side] = node;

    FreeHeader *grandparent;
    FreeHeader *uncle;
    do {
        DEBUG_ASSERT(parent != NULL);
        if (is_black(parent))
            // case I1
            return; // insertion complete

        // we now know parent is red

        grandparent = parent->parent;

        // red and root parent is illegal, but can be fixed in O(1)
        if (grandparent == NULL) {
            // case I4
            DEBUG_ASSERT(parent == rb_root);
            DEBUG_ASSERT(is_red(parent));
            mark_black(parent);
            return;
        }

        // parent is red and has a parent
        side = child_side(parent);
        Side opposite = 1 - side;
        uncle = grandparent->children[opposite];

        // check if we can fix by rotation
        if (uncle == NULL || is_black(uncle)) {
            // case I56
            DEBUG_ASSERT(is_red(parent));
            if (node == parent->children[opposite]) {
                // red parent, black uncle, and `node` is an interior child
                // rotate to make node an outer child
                // (parent is not the root)
                rb_rotate(parent, side);
                node = parent;
                parent = grandparent->children[side];
            }
            // case I6
            // red parent, black uncle, and `node` is an exterior child
            DEBUG_ASSERT(is_red(parent));
            rb_rotate(grandparent, opposite);
            mark_black(parent);
            mark_red(grandparent);
            return;
        }

        // case I2
        // parent and uncle are both red
        // make them black now and the grandparent red

        DEBUG_ASSERT(is_red(parent));
        DEBUG_ASSERT(is_red(uncle));
        mark_black(parent);
        mark_black(uncle);
        mark_red(grandparent);
        node = grandparent;
        parent = node->parent;
    } while (node->parent != NULL);

    // node is now the root and red and the tree is fixed up
}

static void rb_delete(FreeHeader *const node) {
    FreeHeader *const left_child = node->children[LEFT];
    FreeHeader *const right_child = node->children[RIGHT];

    // If the node is a lonely root, we can just delete the tree.
    if (node == rb_root && left_child == NULL && right_child == NULL) {
        rb_root = NULL;
        return;
    }

    // If the node has a single child colored red, we can paint that child black and replace
    // the node with its child.
    if (left_child == NULL && right_child != NULL) {
        DEBUG_ASSERT(is_red(right_child));
        if (node == rb_root)
            rb_root = right_child;
        else
            node->parent->children[child_side(node)] = right_child;
        right_child->parent = node->parent;
        mark_black(right_child);

        return;
    } else if (left_child != NULL && right_child == NULL) {
        DEBUG_ASSERT(is_red(left_child));
        if (node == rb_root)
            rb_root = left_child;
        else
            node->parent->children[child_side(node)] = left_child;
        left_child->parent = node->parent;
        mark_black(left_child);

        return;
    }

    // node either has two real children or two nil children
    DEBUG_ASSERT((node->children[LEFT] != NULL && node->children[RIGHT] != NULL) ||
                 (node->children[LEFT] == NULL && node->children[RIGHT] == NULL));

    if (node->children[RIGHT] != NULL) {
        // If we have two non-NIL children, replace ourselves with our in-order successor
        FreeHeader *successor = right_child;
        while (successor->children[LEFT] != NULL)
            successor = successor->children[LEFT];

        rb_delete(successor);

        // `successor` no longer exists, so `node` can safely be replaced by it
        if (node != rb_root)
            node->parent->children[child_side(node)] = successor;
        else
            rb_root = successor;
        successor->parent = node->parent;

        successor->children[LEFT] = node->children[LEFT];
        if (node->children[LEFT] != NULL)
            node->children[LEFT]->parent = successor;

        if (successor != node->children[RIGHT]) {
            successor->children[RIGHT] = node->children[RIGHT];
            if (node->children[RIGHT] != NULL)
                node->children[RIGHT]->parent = successor;
        }
        successor->size_flags = (successor->size_flags & ~RED) | (node->size_flags & RED);
        return;
    } else if (is_red(node)) {
        // A red node with no children can be silently removed from its parent's memory with no
        // trouble.
        node->parent->children[child_side(node)] = NULL;
        return;
    }

    // `node` is a black node with two black `NIL` children.
    // this is the hardest case - deleting it requires going up the tree and fixing everything.
    rb_delete_hard(node);
}

static inline void rb_delete_hard(FreeHeader *node) {
    DEBUG_ASSERT(node->children[LEFT] == NULL);
    DEBUG_ASSERT(node->children[RIGHT] == NULL);
    DEBUG_ASSERT(node->parent != NULL);
    DEBUG_ASSERT(is_black(node));

    Side side = child_side(node);

    FreeHeader *parent = node->parent;
    parent->children[side] = NULL;

    FreeHeader *sibling;
    FreeHeader *distant_nephew;
    FreeHeader *close_nephew;

    goto RbDeleteLoopStart;
    do {
        side = child_side(node);
    RbDeleteLoopStart:;
        sibling = parent->children[1 - side];
        distant_nephew = sibling->children[1 - side];
        close_nephew = sibling->children[side];

        if (is_red(sibling))
            goto RbDeleteRedSibling; // D3

        if (distant_nephew != NULL && is_red(distant_nephew))
            goto RbDeleteRedDistantNephew; // D6

        if (close_nephew != NULL && is_red(close_nephew))
            goto RbDeleteRedCloseNephew; // D5

        if (parent != NULL && is_red(parent))
            goto RbDeleteBlackFamily; // D4

        DEBUG_ASSERT(parent == NULL || is_black(parent));
        DEBUG_ASSERT(sibling == NULL || is_black(sibling));
        DEBUG_ASSERT(close_nephew == NULL || is_black(close_nephew));
        DEBUG_ASSERT(distant_nephew == NULL || is_black(distant_nephew));

        mark_red(sibling);
        node = parent;
        parent = node->parent;

    } while (parent != NULL);

    // parent is null - we are done
    return;

RbDeleteRedSibling: // D3
    DEBUG_ASSERT(parent == NULL || is_black(parent));
    DEBUG_ASSERT(sibling != NULL && is_red(sibling));
    DEBUG_ASSERT(close_nephew == NULL || is_black(close_nephew));
    DEBUG_ASSERT(distant_nephew == NULL || is_black(distant_nephew));
    rb_rotate(parent, side);
    mark_red(parent);
    mark_black(sibling);
    sibling = close_nephew;
    // now: parent red and sibling black
    distant_nephew = sibling->children[1 - side];
    if (distant_nephew != NULL && is_red(distant_nephew))
        goto RbDeleteRedDistantNephew;

    close_nephew = sibling->children[side];
    if (close_nephew != NULL && is_red(close_nephew))
        goto RbDeleteRedCloseNephew;
    // fall through to deleting black family
RbDeleteBlackFamily: // D4
    DEBUG_ASSERT(parent != NULL && is_red(parent));
    DEBUG_ASSERT(sibling == NULL || is_black(sibling));
    DEBUG_ASSERT(close_nephew == NULL || is_black(close_nephew));
    DEBUG_ASSERT(distant_nephew == NULL || is_black(distant_nephew));

    mark_red(sibling);
    mark_black(parent);
    return;

RbDeleteRedCloseNephew: // D5
    DEBUG_ASSERT(parent == NULL || is_black(sibling));
    DEBUG_ASSERT(close_nephew != NULL && is_red(close_nephew));
    DEBUG_ASSERT(distant_nephew == NULL || is_black(distant_nephew));

    rb_rotate(sibling, 1 - side);
    mark_red(sibling);
    mark_black(close_nephew);
    distant_nephew = sibling;
    sibling = close_nephew;

RbDeleteRedDistantNephew: // D6
    DEBUG_ASSERT(sibling == NULL || is_black(sibling));
    DEBUG_ASSERT(distant_nephew != NULL && is_red(distant_nephew));

    rb_rotate(parent, side);
    sibling->size_flags = (sibling->size_flags & ~RED) | (parent->size_flags & RED);
    mark_black(parent);
    mark_black(distant_nephew);
}

static inline void rb_rotate(FreeHeader *const parent, const Side side) {
    const Side opposite = 1 - side;
    FreeHeader *grandparent = parent->parent;
    FreeHeader *sibling = parent->children[opposite];
    DEBUG_ASSERT(sibling != NULL);
    FreeHeader *child = sibling->children[side];

    // make `parent` steal the sibling's child
    parent->children[opposite] = child;
    if (child != NULL)
        child->parent = parent;

    // make the sibling have the parent as its new child as replacement for its stolen child
    sibling->children[side] = parent;
    parent->parent = sibling;
    sibling->parent = grandparent;

    // make the grandparent point down to the sibling
    if (grandparent != NULL)
        grandparent->children[parent == grandparent->children[RIGHT] ? RIGHT : LEFT] = sibling;
    else // if there is no grandparent, overwrite the root
        rb_root = sibling;
}

static size_t rb_check_subtree(const FreeHeader *node) {
    if (node == NULL)
        return 1; // nil is black

    DEBUG_PRINTF("check node %p (size %zu)\n", node, data_size(node->size_flags));

    if (node == rb_root)
        DEBUG_ASSERT(node->parent == NULL);

    if (is_red(node)) {
        DEBUG_ASSERT(node->children[LEFT] == NULL || is_black(node->children[LEFT]));
        DEBUG_ASSERT(node->children[RIGHT] == NULL || is_black(node->children[RIGHT]));
    }

    if (node->children[LEFT] != NULL) {
        DEBUG_ASSERT(node->children[LEFT]->parent == node);
        DEBUG_ASSERT(data_size(node->children[LEFT]->size_flags) <= data_size(node->size_flags));
    }
    if (node->children[RIGHT] != NULL) {
        DEBUG_ASSERT(node->children[RIGHT]->parent == node);
        DEBUG_ASSERT(data_size(node->children[RIGHT]->size_flags) >= data_size(node->size_flags));
    }

    DEBUG_ASSERT(first_block <= (BlockHeader *)node && (BlockHeader *)node <= last_block);
    DEBUG_ASSERT((node->size_flags & 0b01) == 0 &&); // not allocated and word aligned

    const size_t left_black_depth = rb_check_subtree(node->children[LEFT]);
    const size_t right_black_depth = rb_check_subtree(node->children[RIGHT]);
    DEBUG_ASSERT(left_black_depth == right_black_depth);

    return is_black(node) ? left_black_depth + 1 : right_black_depth;
}

static inline size_t adjust_size(size_t request_size) {
    // mask out hanging bits, then round up
    return request_size <= MIN_PAYLOAD_SIZE ? MIN_PAYLOAD_SIZE
                                            : ((request_size - 1) & ~(WSIZE - 1)) + WSIZE;
}

static void check_heap() {
    // Validate first and last block
    DEBUG_ASSERT(first_block == mem_heap_lo());
    if (last_block == NULL)
        DEBUG_ASSERT(mem_heap_lo() == mem_heap_hi() + 1);
    else
        DEBUG_ASSERT((char *)find_footer(last_block) + sizeof(Footer) == mem_heap_hi() + 1);

    // Check the properties of the tree
    if (rb_root != NULL)
        DEBUG_ASSERT(rb_root->parent == NULL);

    rb_check_subtree(rb_root);

    // Go through every block on the heap and validate it
    for (const BlockHeader *bh = first_block; bh <= last_block; bh = next_block(bh)) {
        if (is_allocated(bh->size_flags))
            DEBUG_ASSERT((bh->size_flags & ~0b11) == (find_footer(bh)->size_flags & ~0b11));
        else
            DEBUG_ASSERT((bh->size_flags & ~0b10) ==
                         find_footer_free((FreeHeader *)bh)->size_flags);
        DEBUG_ASSERT(data_size(bh->size_flags) >= MIN_PAYLOAD_SIZE);
    }
}