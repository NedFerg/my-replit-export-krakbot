import { Router, type IRouter } from "express";
import healthRouter from "./health";
import tradesRouter from "./trades";

const router: IRouter = Router();

router.use(healthRouter);
router.use(tradesRouter);

export default router;
