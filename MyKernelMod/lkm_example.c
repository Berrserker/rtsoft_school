#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Yago Sevatarion");
MODULE_DESCRIPTION("A simple example Linux modul.");
MODULE_VERSION("0.01");

static int __init lkm_example_init(void) {
<<<<<<< HEAD
  printk(KERN_INFO "Hello, World!\n");
=======
  printk(KERN_INFO “Hello, World!\n”);
>>>>>>> parent of 13c926e... stg
  return 0;
}
static void __exit lkm_example_exit(void) {
  printk(KERN_INFO "Goodbye, World!\n");
}

module_init(lkm_example_init);
module_exit(lkm_example_exit);
