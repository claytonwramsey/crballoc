# `crballoc`: A memory allocator based on a red-black tree

This is an implementation of a memory allocator based on red-black trees.
I originally wrote this as a solution for the CS:APP `malloc` assignment but since have adapted it
for the C memory allocation API.

Free blocks are all stored in one large red-black tree, sorted by the size of the data segment of
the block.
This guarantees the best-fitting free block will be found for all allocation requests.
However, allocations and frees are performed in $O(\log n)$, where $n$ is the number of free blocks
in the heap.

I would only recommend this allocator in cases where you have a relatively small number of free
blocks, all of which are large, and which require a minimal memory overhead, and where your
application is single-threaded.
In other words, this allocator isn't very useful.

## File structure

Two implementations are contained here:

- [csapp.c](csapp.c): The original implementation for the CS:APP assignment.
  Due to copyright, I have not included the supporting files for the assignment - go find them on
  your own.
- [crballoc.c](crballoc.c): The implementation for the C memory allocation API.

## License

This software is licensed to you under the GNU GPLv3.
For more information, refer to [LICENSE.md](LICENSE.md).
